"""Cascade stage: the classification tree each selector's descriptors grow.

Ported from cascade_by_method.py. The target is binned into classes
(class_edges). For each selection method, one classification tree is grown
on the full dataset from that method's selected descriptors only (its
pool), to compare which descriptors each pool leads the tree to split on.
By default the methods are the tree-based selectors the selection stage
ran (BorutaShap and XGBoost in the paper).

Each branch keeps splitting while its majority-class fraction is below
purity_stop, and becomes a leaf once it reaches it, at max_depth, or when it
has fewer rows than twice the minimum leaf, max(min_leaf, min_leaf_frac x
rows). Every split is the best depth-1 CART split (Gini) over the whole
pool, so no descriptor is forced.

In the figures each split is a histogram of its descriptor over the rows
reaching it, with the threshold, and each leaf a pie of the classes with
its size and mean target. Every layout setting of the script is in
cascade.figure, with the script's values as defaults. The layout is fitted
in inches: with auto_fit, the script's estimate sets the starting spacing,
then the drawn figure is measured and the spacing widened wherever two
elements are still closer than audit_pad_in. An audit after each figure
prints anything that still touches.

run() grows the trees and writes the tables, and saves the trees with the
rows they need in plot_data/. plot() draws the figures from those, so
`fsphysics plot` restyles them without growing the trees again.

Differences from cascade_by_method.py, all deliberate:
  * The pools are this run's selection-stage sets, and the methods are a
    setting; with more than two, the summary lists the split descriptors
    common to all of them.
  * class_edges default to auto, the target's tertiles at 3 significant
    figures, for a new dataset; the paper's config sets -3 and -2 eV.
  * The units, the legend title and the mean column follow
    data.target_unit, data.target_label and data.target (the script: eV,
    "Solvation energy", mean_Total_eV).
  * The symbol for the mean on the leaves, bold text and the class colours
    are settings (figure.mean_symbol, figure.bold, figure.class_colors).
    mean_symbol auto gives the script's <E> for a target in an energy unit
    and "mean" otherwise.
  * A method that selected no descriptors is skipped with a note (the
    script stopped with an error), and a tree without a split is not drawn.
  * auto_fit also measures the drawn figure, as above. The script's
    estimate spaces neighbours on the same row by estimated text widths and
    leaves out the labels on the arrows, which could touch the descriptor
    names and each other; the measurement catches those and anything else
    the estimate misses. A figure the estimate already spaced cleanly is
    drawn exactly as the script drew it; one that is widened says so
    ("[fit] ... widened").
  * The audit checks every pair of elements of different nodes, the arrows
    (one running through another node), the legend and the panel label;
    the script's checked thirteen kinds of pair, not a pie against a
    histogram, and counted the labels of ticks beyond the axis, which are
    never drawn. With auto_fit off, it reports what overlaps.
  * With align_leaves_bottom, an elbow to a leaf drawn on the bottom row
    turns in the gap under its parent and drops in the leaf's own column;
    the script turned partway down the long drop, through the nodes in
    between.
  * A legend in the top left corner goes under a panel label, not on it.
  * The summary is cascade_summary.txt (the script's
    cascade_method_comparison.txt), with the classes and the stopping rule
    above the script's lines.
"""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.transforms import Bbox
from matplotlib.path import Path as MplPath
from sklearn.tree import DecisionTreeClassifier

from . import outputs, style

TREE_METHODS = ("borutashap", "xgboost")   # methods: auto -> these, if run
_CHOICES = {"edge_style": ("straight", "elbow"),
            "hist_mode": ("plain", "split", "stacked"),
            "arrow_label_mode": ("value", "symbol", "none")}


# =============================================================================
# Settings and classes
# =============================================================================
def _figure_settings(cfg):
    """cascade.figure as attributes, after checking the choices."""
    f = cfg["cascade"]["figure"]
    for key, allowed in _CHOICES.items():
        if f[key] not in allowed:
            raise ValueError(f"cascade.figure.{key}: {f[key]!r} (use "
                             f"{' | '.join(allowed)})")
    if len(f["margins"]) != 4:
        raise ValueError("cascade.figure.margins: four fractions (left, "
                         "right, top, bottom)")
    if not f["save_formats"]:
        raise ValueError("cascade.figure.save_formats: at least one format")
    return SimpleNamespace(**f)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_tree_settings(c):
    tests = {
        "purity_stop": (lambda v: 0 < v <= 1, "a fraction above 0, up to 1"),
        "max_depth": (lambda v: v == int(v) and v >= 0,
                      "a whole number, 0 or more"),
        "min_leaf": (lambda v: v == int(v) and v >= 1,
                     "a whole number, 1 or more"),
        "min_leaf_frac": (lambda v: 0 <= v < 0.5,
                          "a fraction from 0 up to, not including, 0.5"),
    }
    for key, (ok, need) in tests.items():
        if not (_number(c[key]) and ok(c[key])):
            raise ValueError(f"cascade.{key}: {c[key]!r} (use {need})")


def _methods(setting, cfg, sets):
    """The methods to grow trees for, in the selection stage's order."""
    if setting == "auto":
        return [m for m in cfg["selection"]["methods"]
                if m in TREE_METHODS and m in sets.columns]
    methods = list(setting)
    missing = [m for m in methods if m not in sets.columns]
    if missing:
        raise KeyError(f"cascade.methods: {missing} are not in the selection "
                       f"stage's selected_feature_sets.csv "
                       f"({list(sets.columns)}); add them to "
                       f"selection.methods and run the selection stage "
                       f"again.")
    return methods


def class_edges(setting, y):
    """The class edges in target units, increasing. auto: the tertiles of
    the target, at 3 significant figures (more if needed to keep them
    apart)."""
    if setting == "auto":
        q = np.quantile(np.asarray(y, dtype=float), [1 / 3, 2 / 3])
        for digits in range(3, 18):
            edges = [float(f"{v:.{digits}g}") for v in q]
            if edges[0] < edges[1]:
                return edges
        raise ValueError("The target has too few distinct values for "
                         "automatic classes; set cascade.class_edges.")
    if (not isinstance(setting, (list, tuple)) or not setting
            or not all(_number(e) for e in setting)
            or len(set(setting)) != len(setting)):
        raise ValueError(f"cascade.class_edges: {setting!r} (use a list of "
                         f"one or more different numbers, or auto)")
    return sorted(float(e) for e in setting)


def make_classes(y, edges, unit):
    """Class of each row (a value equal to an edge goes to the class above
    it) and the class labels, lowest first."""
    codes = np.digitize(y, edges)
    u = f" {unit}" if unit else ""
    labels = []
    for i in range(len(edges) + 1):
        if i == 0:
            labels.append(f"< {edges[0]:g}{u}")
        elif i == len(edges):
            labels.append(f"> {edges[-1]:g}{u}")
        else:
            labels.append(f"{edges[i-1]:g} to {edges[i]:g}{u}")
    return codes, labels


def _class_colors(S, k):
    if S.class_colors == "auto":
        return style.class_colors(k)
    colors = list(S.class_colors)
    if len(colors) != k:
        raise ValueError(f"cascade.figure.class_colors: {len(colors)} "
                         f"colours for {k} classes")
    return colors


# =============================================================================
# Adaptive tree
# =============================================================================
class Node:
    __slots__ = ("rows", "depth", "feature", "threshold", "left", "right")

    def __init__(self, rows, depth):
        self.rows = rows
        self.depth = depth
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None

    @property
    def is_leaf(self):
        return self.left is None


def _purity(codes, rows, k):
    c = np.bincount(codes[rows], minlength=k)
    return c.max() / c.sum()


def grow(Xf, codes, rows, depth, k, min_leaf, rule):
    """rule = (purity_stop, max_depth, random_state)."""
    purity_stop, depth_cap, seed = rule
    node = Node(rows, depth)
    if (depth >= depth_cap or len(rows) < 2 * min_leaf
            or _purity(codes, rows, k) >= purity_stop):
        return node                                   # stop: leaf

    stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=min_leaf,
                                   random_state=seed)
    stump.fit(Xf.iloc[rows], codes[rows])
    if stump.tree_.node_count < 3:                    # no valid split
        return node

    fname = Xf.columns[stump.tree_.feature[0]]
    thr = float(stump.tree_.threshold[0])
    mask = Xf[fname].values[rows] <= thr
    Lr, Rr = rows[mask], rows[~mask]
    if len(Lr) < min_leaf or len(Rr) < min_leaf:
        return node

    node.feature, node.threshold = fname, thr
    node.left = grow(Xf, codes, Lr, depth + 1, k, min_leaf, rule)
    node.right = grow(Xf, codes, Rr, depth + 1, k, min_leaf, rule)
    return node


def leaves_in_order(node, acc):
    if node.is_leaf:
        acc.append(node)
    else:
        leaves_in_order(node.left, acc)
        leaves_in_order(node.right, acc)


def max_depth(root):
    acc = []
    leaves_in_order(root, acc)
    return max(n.depth for n in acc)


def _save_tree(root, n_rows):
    """The tree as a table (pre-order ids) and the leaf id of every row."""
    nodes, leaf_of = [], np.full(n_rows, -1, dtype=int)

    def walk(n, parent, side):
        i = len(nodes)
        nodes.append({"node": i, "parent": parent, "side": side,
                      "depth": n.depth,
                      "feature": "" if n.is_leaf else n.feature,
                      "threshold": np.nan if n.is_leaf else n.threshold,
                      "n": len(n.rows)})
        if n.is_leaf:
            leaf_of[n.rows] = i
        else:
            walk(n.left, i, "left")
            walk(n.right, i, "right")

    walk(root, -1, "root")
    return pd.DataFrame(nodes), leaf_of


def _load_tree(nodes, leaf_of):
    """Rebuild a saved tree; each node gets its rows in increasing order,
    as when it was grown."""
    objs = {}
    for r in nodes.itertuples(index=False):
        n = Node(None, int(r.depth))
        if r.feature != "":
            n.feature, n.threshold = r.feature, float(r.threshold)
        objs[int(r.node)] = n
        if r.side == "left":
            objs[int(r.parent)].left = n
        elif r.side == "right":
            objs[int(r.parent)].right = n
    ids = {id(n): i for i, n in objs.items()}

    def fill(n):
        if n.is_leaf:
            n.rows = np.flatnonzero(leaf_of == ids[id(n)])
        else:
            fill(n.left)
            fill(n.right)
            n.rows = np.sort(np.concatenate([n.left.rows, n.right.rows]))

    fill(objs[0])
    return objs[0]


# =============================================================================
# Geometry
# =============================================================================
def _fs(x, S):
    return x * S.font_scale


def _luminance(rgba):
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _tick_drop_in(S):
    if S.tick_drop_in is not None:
        return S.tick_drop_in
    # tick mark (3.5pt) + tick pad (1.5pt) + one line of tick text
    return (3.5 + 1.5 + 1.30 * _fs(S.fs_tick, S)) / 72.0 + S.text_gap_in


def _label_drop_in(S):
    if S.label_drop_in is not None:
        return S.label_drop_in
    return (1.35 * _fs(S.fs_feature, S)) / 72.0 + S.text_gap_in


def positions(root, S):
    """Return (pos, n_leaves, span_units).

    Leaf columns are variable-width: a leaf that terminates early gets a
    narrower column, which pulls it (and everything outboard of it) inward.
    span_units is the total width in column units and drives the canvas
    width, so squeezing removes canvas rather than redistributing it.
    """
    order = []
    leaves_in_order(root, order)
    md = max(n.depth for n in order)

    widths = []
    for n in order:
        if S.leaf_squeeze <= 0 or md == 0:
            widths.append(1.0)
        else:
            deficit = (md - n.depth) / md          # 0 = deepest, 1 = shallowest
            widths.append(max(S.min_leaf_slot, 1.0 - S.leaf_squeeze * deficit))

    # Only squeeze a leaf with no neighbour drawn on its own row: narrowing
    # one that has a neighbour cannot bring the two closer (the fitted pitch
    # grows back, and widens every other column with it).
    row_of = ((lambda n: md) if S.align_leaves_bottom
              else (lambda n: n.depth))
    for i, n in enumerate(order):
        same_row = ((i > 0 and row_of(order[i - 1]) == row_of(n))
                    or (i + 1 < len(order)
                        and row_of(order[i + 1]) == row_of(n)))
        if same_row:
            widths[i] = 1.0
    span_units = float(sum(widths))

    leaf_x, run = {}, 0.0
    for n, w in zip(order, widths):
        leaf_x[id(n)] = run + w / 2.0 - 0.5        # fx() adds the +0.5 back
        run += w

    pos = {}

    def place(n):
        if n.is_leaf:
            row = md if S.align_leaves_bottom else n.depth
            pos[id(n)] = (leaf_x[id(n)], row)
            return leaf_x[id(n)]
        xl = place(n.left)
        xr = place(n.right)
        pos[id(n)] = ((xl + xr) / 2.0, n.depth)
        return (xl + xr) / 2.0

    place(root)
    return pos, len(order), span_units


def _plain(text):
    """Plain stand-in for a label that may hold mathtext, for widths:
    "$\\langle E \\rangle$" -> "xEx", as long as "<E>"."""
    if "$" not in text:
        return text
    return re.sub(r"\\[A-Za-z]+", "x", text.replace("$", "")).replace(" ", "")


ENERGY_UNITS = {"ev", "mev", "kcal/mol", "kj/mol", "ha", "hartree", "eh",
                "ry", "j/mol"}


def _mean_symbol(S, unit):
    """figure.mean_symbol; auto = <E> for an energy target, else "mean"."""
    if S.mean_symbol != "auto":
        return S.mean_symbol
    energy = (unit or "").strip().lower() in ENERGY_UNITS
    return "$\\langle E \\rangle$" if energy else "mean"


def _mean_text(symbol, value, S, unit):
    """The second caption line under a pie: "<E> = -4.25 eV"."""
    return (f"{symbol} = {value}"
            + (f" {unit}" if S.leaf_text_units and unit else ""))


def _text_w_in(txt, fs, S):
    """Rough rendered width of a bold DejaVu Sans string, in inches."""
    return len(txt) * 0.62 * _fs(fs, S) / 72.0


def _half_width_in(node, hw_in, pie_in, S, unit):
    """Horizontal half-extent of a node including the text under it."""
    if node.is_leaf:
        cap = max(_text_w_in(f"n = {len(node.rows)}", S.fs_leaf_text, S),
                  _text_w_in(_mean_text(_plain(_mean_symbol(S, unit)),
                                        "-0.00", S, unit),
                             S.fs_leaf_text, S))
        return max(pie_in, cap) / 2.0
    return max(hw_in, _text_w_in(style.short(node.feature), S.fs_feature,
                                 S)) / 2.0


def _fit_pitches(root, pos, hw_in, hh_in, pie_in, tick_drop, label_drop, S,
                 unit):
    """Smallest (col_pitch, row_pitch) in inches that keeps the layout legal.

    Row: a parent's label block hangs below its centre and a child's glyph
    rises above its own; a smaller row pitch would put the child's top edge
    above the parent's anchor and force the arrow to point upward.
    Column: for every pair of neighbours drawn on the same row, the pitch
    must cover both half-extents plus the audit clearance.
    """
    row_need = (hh_in / 2 + tick_drop + label_drop
                + max(pie_in / 2 + S.leaf_top_gap_in,
                      hh_in / 2 + S.node_top_gap_in)
                + S.min_edge_len_in * 1.08)    # slack, so the guard in
    #                                            draw() clears, not ties

    nodes = []

    def collect(n):
        nodes.append(n)
        if not n.is_leaf:
            collect(n.left)
            collect(n.right)
    collect(root)

    rows = {}
    for n in nodes:
        x, row = pos[id(n)]
        rows.setdefault(row, []).append(
            (x, _half_width_in(n, hw_in, pie_in, S, unit), n))

    col_need, why = 0.0, None
    for rownum, row in rows.items():
        row.sort(key=lambda t: (t[0], t[1]))
        for (xa, ha, na), (xb, hb, nb) in zip(row, row[1:]):
            dx = xb - xa
            if dx > 1e-9:
                need = (ha + hb + 2 * S.audit_pad_in) / dx
                if need > col_need:
                    col_need = need
                    why = (rownum,
                           "pie" if na.is_leaf else style.short(na.feature),
                           "pie" if nb.is_leaf else style.short(nb.feature),
                           dx)
    return col_need, row_need, why


def _elbow_path(p0, p1, frac, rx, ry):
    """Orthogonal down / across / down connector, optionally round-cornered."""
    x0, y0 = p0
    x1, y1 = p1
    if abs(x1 - x0) < 1e-9:
        return MplPath([(x0, y0), (x1, y1)], [MplPath.MOVETO, MplPath.LINETO])

    ym = y0 + frac * (y1 - y0)          # y decreases going down the figure
    sx = 1.0 if x1 > x0 else -1.0
    rx = min(rx, abs(x1 - x0) / 2.0)
    ry = min(ry, abs(y0 - ym) * 0.9, abs(ym - y1) * 0.9)

    if rx <= 1e-6 or ry <= 1e-6:
        return MplPath([(x0, y0), (x0, ym), (x1, ym), (x1, y1)],
                       [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
                        MplPath.LINETO])

    verts = [(x0, y0), (x0, ym + ry), (x0, ym), (x0 + sx * rx, ym),
             (x1 - sx * rx, ym), (x1, ym), (x1, ym - ry), (x1, y1)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO]
    return MplPath(verts, codes)


def _elbow_path_side(p0, p1, rx, ry):
    """Down from the parent, then across into the side of the child."""
    x0, y0 = p0
    x1, y1 = p1
    sx = 1.0 if x1 > x0 else -1.0
    rx = min(rx, abs(x1 - x0) * 0.9)
    ry = min(ry, abs(y0 - y1) * 0.9)
    if rx <= 1e-6 or ry <= 1e-6:
        return MplPath([(x0, y0), (x0, y1), (x1, y1)],
                       [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO])
    verts = [(x0, y0), (x0, y1 + ry), (x0, y1), (x0 + sx * rx, y1), (x1, y1)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO]
    return MplPath(verts, codes)


def _edge_label(thr, side, S):
    if S.arrow_label_mode == "none":
        return None
    if S.arrow_label_mode == "symbol":
        return "≤" if side == "left" else ">"
    return (f"≤ {thr:.4g}" if side == "left" else f"> {thr:.4g}")


# =============================================================================
# Drawing
# =============================================================================
_MAX_REFITS = 30        # auto_fit: most redraws while widening the spacing


def draw(root, Xf, y, codes, class_labels, colors, out_base, panel, S, unit,
         legend_title):
    """Draw one cascade, save it in every format, and return
    (W, H, col_pitch, row_pitch) in inches.

    With auto_fit the pitches start from the script's estimate (col_inches
    and row_inches as floors). The drawn figure is then measured, and while
    any two elements of different nodes are closer than audit_pad_in, the
    pitch that separates them at the smaller cost is widened and the figure
    drawn again. A layout the estimate already spaced cleanly is drawn
    exactly as the script drew it.
    """
    depth = max_depth(root)
    k = len(class_labels)
    weight = "bold" if S.bold else "normal"

    LM, RM, TM, BM = S.margins
    span_x, span_y = RM - LM, TM - BM

    hw_in = S.hist_w_in * S.hist_scale
    hh_in = S.hist_h_in * S.hist_scale
    pie_in = S.pie_in * S.pie_scale
    tick_drop_in, label_drop_in = _tick_drop_in(S), _label_drop_in(S)

    pos, n_leaves, span_units = positions(root, S)
    need_col, need_row, why = _fit_pitches(root, pos, hw_in, hh_in, pie_in,
                                           tick_drop_in, label_drop_in, S,
                                           unit)
    col_pitch = max(S.col_inches, need_col) if S.auto_fit else S.col_inches
    row_pitch = max(S.row_inches, need_row) if S.auto_fit else S.row_inches
    if S.auto_fit and why is not None and col_pitch > S.col_inches + 1e-6:
        print(f"  [fit] column pitch {col_pitch:.2f} in is set by "
              f"[{why[1]}] next to [{why[2]}] on row {why[0]}, "
              f"{why[3]:.2f} columns apart")

    def build(col_pitch, row_pitch):
        """The figure at these pitches, and its parts for the audit."""
        # pitch is exact: fx/fy space the rows and columns evenly
        W = col_pitch * span_units / span_x
        H = row_pitch * (depth + 1) / span_y

        def fx(x):
            return LM + (x + 0.5) / span_units * span_x

        def fy(d):
            return TM - (d + 0.5) / (depth + 1) * span_y

        # inches -> figure fraction (so every figure shares one visual scale)
        hw, hh = hw_in / W, hh_in / H
        pie_w, pie_h = pie_in / W, pie_in / H
        TICK_DROP, LABEL_DROP = tick_drop_in / H, label_drop_in / H
        rx, ry = S.elbow_radius_in / W, S.elbow_radius_in / H

        fig = plt.figure(figsize=(W, H))
        bg = fig.add_axes([0, 0, 1, 1])
        bg.axis("off")
        bg.set_xlim(0, 1)
        bg.set_ylim(0, 1)

        parts = {"W": W, "H": H, "centers": {}, "hist": [], "pie": [],
                 "feat": [], "cond": [], "leaftext": [], "nodeN": [],
                 "thresh": [], "edges": [], "legend": None, "panel": None}

        def bottom_y(n):    # y where edges LEAVE (below the box and labels)
            return fy(pos[id(n)][1]) - hh / 2 - TICK_DROP - LABEL_DROP

        def top_y(n):       # y where edges ARRIVE
            if n.is_leaf:
                return fy(pos[id(n)][1]) + pie_h / 2 + S.leaf_top_gap_in / H
            return fy(pos[id(n)][1]) + hh / 2 + S.node_top_gap_in / H

        def child_anchor(parent, child):
            """Where the edge lands on the child, and how it is routed: the
            top edge, or, if the parent's label block hangs lower than the
            child's top, the child's near flank, so no edge points up."""
            cxf = fx(pos[id(child)][0])
            if bottom_y(parent) - top_y(child) >= S.min_edge_len_in / H - 1e-9:
                return (cxf, top_y(child)), "top"
            half = (pie_w if child.is_leaf else hw) / 2
            gap = S.node_top_gap_in / W
            toward = -1.0 if pos[id(child)][0] > pos[id(parent)][0] else 1.0
            return (cxf + toward * (half + gap), fy(pos[id(child)][1])), "side"

        def turn_frac(n, child, p0, p1):
            """Where an elbow turns, as a fraction of the drop to the child:
            elbow_drop_frac. A leaf drawn further down than the next row
            (align_leaves_bottom) is reached by turning in the gap below
            the parent instead, so the arrow drops in the leaf's own column
            rather than through the nodes in between."""
            rows = pos[id(child)][1] - pos[id(n)][1]
            if rows <= 1 or abs(p1[1] - p0[1]) < 1e-12:
                return S.elbow_drop_frac
            next_top = (fy(pos[id(n)][1] + 1) + pie_h / 2
                        + S.leaf_top_gap_in / H)
            ym = p0[1] + S.elbow_drop_frac * (next_top - p0[1])
            return (ym - p0[1]) / (p1[1] - p0[1])

        # ---- edges ----------------------------------------------------------
        def walk_edges(n):
            if n.is_leaf:
                return
            x, thr = pos[id(n)][0], n.threshold
            for child, side, ha in ((n.left, "left", "right"),
                                    (n.right, "right", "left")):
                p0 = (fx(x), bottom_y(n))
                p1, entry = child_anchor(n, child)
                lab = _edge_label(thr, side, S)

                if S.edge_style == "elbow" or entry == "side":
                    frac = turn_frac(n, child, p0, p1)
                    path = (_elbow_path_side(p0, p1, rx, ry)
                            if entry == "side"
                            else _elbow_path(p0, p1, frac, rx, ry))
                    parts["edges"].append(((id(n), side), id(n), id(child),
                                           path.vertices))
                    bg.add_patch(FancyArrowPatch(
                        path=path, arrowstyle="-|>",
                        mutation_scale=S.arrow_head_scale, lw=S.edge_lw,
                        color=S.edge_color, joinstyle="round",
                        capstyle="round", zorder=1))
                    if lab:
                        frac_l = 1.0 if entry == "side" else frac
                        ym = p0[1] + frac_l * (p1[1] - p0[1])
                        mx = p0[0] + 0.72 * (p1[0] - p0[0])
                        t = bg.text(mx, ym + 0.09 / H, lab,
                                    fontsize=_fs(S.fs_cond, S), ha="center",
                                    va="bottom", color="black",
                                    fontweight=weight, zorder=2)
                        parts["cond"].append(((id(n), side), t))
                else:
                    parts["edges"].append(((id(n), side), id(n), id(child),
                                           np.array([p0, p1])))
                    bg.add_patch(FancyArrowPatch(
                        p0, p1, arrowstyle="-|>",
                        mutation_scale=S.arrow_head_scale, lw=S.edge_lw,
                        color=S.edge_color, zorder=1))
                    if lab:
                        mx = p0[0] + 0.60 * (p1[0] - p0[0])
                        my = p0[1] + 0.60 * (p1[1] - p0[1])
                        t = bg.text(mx + (0.004 if ha == "left" else -0.004),
                                    my, lab, fontsize=_fs(S.fs_cond, S), ha=ha,
                                    va="center", color="black",
                                    fontweight=weight, zorder=2)
                        parts["cond"].append(((id(n), side), t))
                walk_edges(child)

        walk_edges(root)

        # ---- nodes ----------------------------------------------------------
        def draw_hist(ax, n):
            vals = Xf[n.feature].values[n.rows]
            if S.hist_mode == "stacked":
                groups = [vals[codes[n.rows] == c] for c in range(k)]
                ax.hist(groups, bins=S.hist_bins, stacked=True, color=colors,
                        edgecolor=S.hist_edge_color, linewidth=S.hist_edge_lw)
            elif S.hist_mode == "split":
                cnt, edges = np.histogram(vals, bins=S.hist_bins)
                ctr = 0.5 * (edges[:-1] + edges[1:])
                wid = np.diff(edges)
                lo = ctr <= n.threshold
                ax.bar(ctr[lo], cnt[lo], width=wid[lo],
                       color=S.hist_color_left, edgecolor=S.hist_edge_color,
                       linewidth=S.hist_edge_lw)
                ax.bar(ctr[~lo], cnt[~lo], width=wid[~lo],
                       color=S.hist_color_right, edgecolor=S.hist_edge_color,
                       linewidth=S.hist_edge_lw)
            else:
                ax.hist(vals, bins=S.hist_bins, color=S.hist_color,
                        edgecolor=S.hist_edge_color, linewidth=S.hist_edge_lw)
            return vals

        def walk_nodes(n):
            cx, cy = fx(pos[id(n)][0]), fy(pos[id(n)][1])
            parts["centers"][id(n)] = (cx, cy)
            if not n.is_leaf:
                ax = fig.add_axes([cx - hw / 2, cy - hh / 2, hw, hh],
                                  zorder=3)
                parts["hist"].append((id(n), ax))
                vals = draw_hist(ax, n)
                ax.axvline(n.threshold, color=S.thresh_color, ls="--",
                           lw=S.thresh_lw, zorder=5)
                ax.set_ylim(top=ax.get_ylim()[1] * S.hist_headroom)
                ax.set_yticks([])
                ax.grid(False)
                ax.tick_params(axis="x", labelsize=_fs(S.fs_tick, S),
                               pad=1.5, colors="black")
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                for lab in ax.get_xticklabels():
                    lab.set_fontweight(weight)
                for s in ("top", "right", "left"):
                    ax.spines[s].set_visible(False)

                frac = ((n.threshold - vals.min())
                        / (vals.max() - vals.min() + 1e-12))
                thr_ann = ax.annotate(
                    f"{n.threshold:.4g}", xy=(n.threshold, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(4 if frac < 0.5 else -4, -2),
                    textcoords="offset points", color=S.thresh_color,
                    fontsize=_fs(S.fs_thresh, S), va="top",
                    fontweight=weight, ha="left" if frac < 0.5 else "right",
                    zorder=6)
                if S.show_node_n and _fs(S.fs_node_n, S) > 0:
                    # outside the frame, top-left: keeps the data area clean
                    # and cannot collide with the threshold annotation
                    t_n = bg.text(cx - hw / 2, cy + hh / 2 + 0.05 / H,
                                  f"n = {len(n.rows)}",
                                  fontsize=_fs(S.fs_node_n, S), ha="left",
                                  va="bottom", color="0.3", fontweight=weight,
                                  zorder=4)
                    parts["nodeN"].append((id(n), t_n))
                parts["thresh"].append((id(n), thr_ann))

                t = bg.text(cx, cy - hh / 2 - TICK_DROP,
                            style.short(n.feature), ha="center", va="top",
                            fontsize=_fs(S.fs_feature, S), fontweight=weight,
                            zorder=4)
                parts["feat"].append((id(n), t))
                walk_nodes(n.left)
                walk_nodes(n.right)
            else:
                ax = fig.add_axes([cx - pie_w / 2, cy - pie_h / 2, pie_w,
                                   pie_h], zorder=3)
                parts["pie"].append((id(n), ax))
                counts = np.bincount(codes[n.rows], minlength=k)
                wedges, _, autotexts = ax.pie(
                    counts, colors=colors, startangle=90, counterclock=False,
                    autopct=lambda p: (f"{p:.0f}%" if p >= S.pie_pct_min
                                       else ""),
                    textprops={"fontsize": _fs(S.fs_pie_pct, S),
                               "fontweight": weight},
                    wedgeprops={"edgecolor": "white", "linewidth": 0.6})
                # white on a light wedge is unreadable in print: per wedge
                for w, at in zip(wedges, autotexts):
                    at.set_color("black"
                                 if _luminance(w.get_facecolor()) > 0.55
                                 else "white")
                # ax.pie leaves a 1.25 margin, so the circle fills only 80%
                # of the box; the caption is placed from the CIRCLE edge, in
                # inches, so its spacing does not change with pie_scale.
                circle_bottom = cy - pie_h / 2 * (1.0 / 1.25)
                line_h = (S.leaf_text_leading * _fs(S.fs_leaf_text, S)
                          / 72.0 / H)
                y1 = circle_bottom - S.leaf_text_gap_in / H
                t1 = bg.text(cx, y1, f"n = {int(counts.sum())}",
                             ha="center", va="top",
                             fontsize=_fs(S.fs_leaf_text, S),
                             fontweight=weight, zorder=4)
                t2 = bg.text(cx, y1 - line_h,
                             _mean_text(_mean_symbol(S, unit),
                                        f"{y.values[n.rows].mean():.2f}", S,
                                        unit),
                             ha="center", va="top",
                             fontsize=_fs(S.fs_leaf_text, S), color="black",
                             fontweight=weight, zorder=4)
                parts["leaftext"] += [(id(n), t1), (id(n), t2)]

        walk_nodes(root)

        if panel:
            parts["panel"] = fig.text(0.005, 0.995, panel,
                                      fontsize=_fs(S.panel_label_fs, S),
                                      fontweight=weight, ha="left", va="top")

        loc = S.legend_loc
        if loc == "auto":
            # put it in whichever top corner the tree actually leaves empty
            top_x = [pos[i][0] for i in pos]
            lean = (np.mean(top_x) + 0.5) / span_units
            loc = "upper left" if lean > 0.5 else "upper right"
        below = {}
        if panel and loc == "upper left":
            # the panel label holds that corner: the legend goes under it
            drop = 1.3 * _fs(S.panel_label_fs, S) / 72.0 / H
            below = {"bbox_to_anchor": (0.0, 0.995 - drop),
                     "bbox_transform": fig.transFigure}
        parts["legend"] = fig.legend(
            handles=[Patch(facecolor=c, edgecolor="white") for c in colors],
            labels=class_labels, loc=loc, title=legend_title,
            fontsize=_fs(S.legend_fontsize, S),
            title_fontsize=_fs(S.legend_title_fontsize, S),
            frameon=S.legend_frame, **below)
        return fig, parts

    start = (col_pitch, row_pitch)
    first = found = None
    for attempt in range(_MAX_REFITS + 1):
        fig, parts = build(col_pitch, row_pitch)
        if not (S.audit or S.auto_fit):
            break
        found = measure(fig, parts, S)
        first = first or found
        if not S.auto_fit or attempt == _MAX_REFITS:
            break
        grow = _growth([(a, b) for _, a, b in found["movable"]],
                       found["crossings"], found["size"])
        if grow is None:
            break
        plt.close(fig)
        col_pitch, row_pitch = (_widen(col_pitch, grow[0]),
                                _widen(row_pitch, grow[1]))
    tag = os.path.basename(out_base)
    if (col_pitch, row_pitch) != start:
        what = [f"{name} pitch to {new:.2f} in (from {old:.2f})"
                for name, new, old in (("column", col_pitch, start[0]),
                                       ("row", row_pitch, start[1]))
                if new != old]
        cleared = {}
        for name, _, _ in first["movable"]:
            cleared[name] = cleared.get(name, 0) + 1
        for _, e in first["crossings"]:
            cleared[f"edge-{e[0]}"] = cleared.get(f"edge-{e[0]}", 0) + 1
        print(f"  [fit] {tag}: widened {' and '.join(what)}, to clear "
              + ", ".join(f"{k}={v}" for k, v in cleared.items()))
    if S.audit:
        report(found, tag, S)

    for ext in S.save_formats:
        fig.savefig(f"{out_base}.{ext}", dpi=S.dpi, bbox_inches="tight",
                    pad_inches=S.pad_inches)
    plt.close(fig)
    return parts["W"], parts["H"], col_pitch, row_pitch


def _widen(pitch, factor):
    """A wider pitch, rounded up to 0.01 in (so a printed pitch is exact);
    at least 1% wider, so every redraw makes progress."""
    if factor <= 1.0:
        return pitch
    return np.ceil(pitch * max(factor, 1.01) * 100.0) / 100.0


def _growth(pairs, crossings=(), size=None):
    """(column, row) factors that clear every movable collision, treating
    each element as rigid around the node or label it belongs to: a pair
    is cleared by the pitch needing the smaller factor. The legend and the
    panel label stay in their top corner while the tree spreads, so for
    them the lever is the node's distance from that corner. An arrow
    through an element widens the pitch across the arrow by a step (5%),
    redrawn until it clears. None if nothing needs clearing."""
    col = row = 1.0
    for (dx, dy), _ in crossings:
        if abs(dy) >= abs(dx):
            col = max(col, 1.05)
        else:
            row = max(row, 1.05)
    for a, b in pairs:
        ox = min(a[2].x1, b[2].x1) - max(a[2].x0, b[2].x0)
        oy = min(a[2].y1, b[2].y1) - max(a[2].y0, b[2].y0)
        if size is not None and (a[0] in _ANCHORED or b[0] in _ANCHORED):
            pin, node = (a, b) if a[0] in _ANCHORED else (b, a)
            right = pin[3][0] > size[0] / 2.0
            sx = size[0] - node[3][0] if right else node[3][0]
            sy = size[1] - node[3][1]
        else:
            sx = abs(a[3][0] - b[3][0])
            sy = abs(a[3][1] - b[3][1])
        need_c = 1.0 + ox / sx if sx > 0.5 else np.inf     # pixels
        need_r = 1.0 + oy / sy if sy > 0.5 else np.inf
        if np.isinf(need_c) and np.isinf(need_r):
            continue
        if need_c <= need_r:
            col = max(col, need_c)
        else:
            row = max(row, need_r)
    return (col, row) if col > 1.0 or row > 1.0 else None


# =============================================================================
# Layout audit, after every figure, so size changes are safe
# =============================================================================
# The script's audit: these kinds of pair, counted its way. The package adds
# every other pair of elements that belong to different nodes or labels.
_SCRIPT_KINDS = ["hist-hist", "pie-pie", "leaftext-leaftext", "feat-feat",
                 "feat-tick", "feat-cond", "cond-cond", "cond-hist",
                 "cond-pie", "leaftext-hist", "nodeN-thresh", "nodeN-nodeN",
                 "nodeN-cond"]
_KINDS = ["hist", "pie", "tick", "feat", "thresh", "nodeN", "leaftext", "cond",
          "legend", "panel"]
_ANCHORED = {"legend", "panel"}       # both pinned to a corner of the figure


def measure(fig, parts, S):
    """Every pair of drawn elements closer than audit_pad_in.

    Returns {"counts": {kind: pairs, ...}, "movable": [(a, b), ...]}, the
    counts in the script's order and then the package's added kinds, and
    the colliding pairs that more spacing can separate. An element is
    (kind, owner, box, owner centre), in pixels.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = S.audit_pad_in * fig.dpi / 2.0     # inflate every box by half a pad
    to_px = fig.transFigure.transform

    def bb(t):
        return t.get_window_extent(renderer=r).expanded(1, 1).padded(pad)

    def centre(box):
        return ((box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0)

    els = {kind: [] for kind in _KINDS}
    pie_circles = []
    for owner, ax in parts["hist"]:
        node = to_px(parts["centers"][owner])
        els["hist"].append(("hist", owner,
                            ax.get_window_extent(renderer=r).padded(pad),
                            node))
        els["tick"] += [("tick", owner, bb(t), node)
                        for t in _drawn_ticklabels(ax)]
    for owner, ax in parts["pie"]:
        node = to_px(parts["centers"][owner])
        box = ax.get_window_extent(renderer=r)
        els["pie"].append(("pie", owner, box.padded(pad), node))
        # the visible pie fills 80% of its box (ax.pie's 1.25 margin)
        shrink = 0.1 * box.width
        circle = Bbox.from_extents(box.x0 + shrink, box.y0 + shrink,
                                   box.x1 - shrink, box.y1 - shrink)
        pie_circles.append(("pie", owner, circle.padded(pad), node))
    for kind in ("feat", "thresh", "nodeN", "leaftext"):
        for owner, t in parts[kind]:
            els[kind].append((kind, owner, bb(t),
                              to_px(parts["centers"][owner])))
    for owner, t in parts["cond"]:
        box = bb(t)
        els["cond"].append(("cond", owner, box, centre(box)))
    box = parts["legend"].get_window_extent(renderer=r).padded(pad)
    els["legend"].append(("legend", "legend", box, centre(box)))
    if parts["panel"] is not None:
        box = bb(parts["panel"])
        els["panel"].append(("panel", "panel", box, centre(box)))

    def ov(a, b):
        return not (a.x1 <= b.x0 or b.x1 <= a.x0
                    or a.y1 <= b.y0 or b.y1 <= a.y0)

    hits = {}

    def collect(name, A, B, same_owner_ok):
        found = []
        pairs = ((A[i], A[j]) for i in range(len(A))
                 for j in range(i + 1, len(A))) if A is B else \
            ((a, b) for a in A for b in B)
        for a, b in pairs:
            if a[1] == b[1] and not same_owner_ok:
                continue
            if ov(a[2], b[2]):
                found.append((a, b))
        if found:
            hits[name] = found

    script = set()
    for name in _SCRIPT_KINDS:
        ka, kb = name.split("-")
        script.add(frozenset((ka, kb)))
        # the script compares a pie's two caption lines with other pies'
        # only, and every other pair whatever it belongs to
        collect(name, els[ka], els[kb] if ka != kb else els[ka],
                same_owner_ok=(name != "leaftext-leaftext"))
    for i, ka in enumerate(_KINDS):
        for kb in _KINDS[i:]:
            if frozenset((ka, kb)) in script:
                continue
            A = pie_circles if ka == "pie" else els[ka]
            B = pie_circles if kb == "pie" else els[kb]
            collect(f"{ka}-{kb}", A, B if ka != kb else A,
                    same_owner_ok=False)

    movable = [(name, a, b) for name, found in hits.items()
               for a, b in found
               if a[1] != b[1] and not {a[0], b[0]} <= _ANCHORED]

    # an arrow running through an element of any other node or label
    others = [e for kind in _KINDS for e in
              (pie_circles if kind == "pie" else els[kind])]
    crossings = {}
    for owner, parent, child, verts in parts["edges"]:
        pts = to_px(np.asarray(verts, dtype=float))
        for e in others:
            if e[1] in (parent, child, owner):
                continue
            seg = _segment_through(pts, e[2])
            if seg is not None:
                name = f"edge-{e[0]}"
                crossings.setdefault(name, []).append((seg, e))
    counts = {name: len(found) for name, found in hits.items()}
    counts.update({name: len(found) for name, found in crossings.items()})
    return {"counts": counts, "movable": movable,
            "crossings": [c for found in crossings.values() for c in found],
            "size": (fig.bbox.width, fig.bbox.height)}


def _segment_through(pts, box):
    """The first segment of a polyline that passes through a box, as
    (dx, dy) in pixels, or None."""
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        # clip the segment to the box (Liang-Barsky)
        t0, t1 = 0.0, 1.0
        dx, dy = x1 - x0, y1 - y0
        inside = True
        for p, q in ((-dx, x0 - box.x0), (dx, box.x1 - x0),
                     (-dy, y0 - box.y0), (dy, box.y1 - y0)):
            if p == 0:
                if q < 0:
                    inside = False
                    break
                continue
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                inside = False
                break
        if inside and t1 - t0 > 1e-9:
            return (dx, dy)
    return None


def _drawn_ticklabels(ax):
    """The x tick labels actually drawn: the locator also makes ticks
    beyond the axis limits, whose labels have text and a position but are
    not drawn (the script's audit counted them)."""
    lo, hi = sorted(ax.get_xlim())
    tol = 1e-9 * (hi - lo)
    return [t for t in ax.get_xticklabels()
            if t.get_text() and lo - tol <= t.get_position()[0] <= hi + tol]


def report(found, tag, S):
    """Print the audit line: the script's, with the added kinds after."""
    counts = found["counts"]
    if not counts:
        print(f"  [audit] {tag}: clear")
        return
    hits = set(counts)
    tips = []
    if S.auto_fit and not found["movable"] and not found["crossings"]:
        # auto_fit already widened what spacing can clear
        tips.append("these do not clear with more spacing: check the font "
                    "sizes and tick_drop_in/label_drop_in")
    else:
        # which setting fixes it: horizontal crowding -> wider columns,
        # vertical crowding -> taller rows
        horiz = {"hist-hist", "pie-pie", "leaftext-leaftext", "feat-feat",
                 "nodeN-nodeN"}
        vert = {"feat-tick", "cond-hist", "cond-pie", "leaftext-hist",
                "nodeN-thresh"}
        if hits & horiz:
            tips.append("raise cascade.figure.col_inches (or lower "
                        "hist_scale/pie_scale)")
        if hits & vert:
            tips.append("raise cascade.figure.row_inches (or lower "
                        "hist_scale)")
        if hits & {"feat-cond", "nodeN-cond"}:
            tips.append("raise cascade.figure.col_inches, or set "
                        "arrow_label_mode: symbol")
        if not S.auto_fit and hits - set(_SCRIPT_KINDS):
            tips.append("turn cascade.figure.auto_fit on, which widens the "
                        "spacing until nothing touches")
    if "legend-panel" in hits:
        tips.append("move the legend away from the panel label "
                    "(legend_loc)")
    print(f"  [audit] {tag}: COLLISIONS -> "
          + ", ".join(f"{k}={v}" for k, v in counts.items())
          + ("\n           fix: " + "; ".join(sorted(set(tips)))
             if tips else ""))


# =============================================================================
# Tables
# =============================================================================
def export(root, y, codes, class_labels, out_dir, method, target):
    """cascade_<method>_thresholds.csv and _leaves.csv; returns the split
    descriptors in tree order."""
    k = len(class_labels)
    thr_rows, leaf_rows = [], []

    def walk(n, conds):
        if n.is_leaf:
            c = np.bincount(codes[n.rows], minlength=k)
            rec = {"leaf_depth": n.depth,
                   "rule": " AND ".join(conds) if conds else "(root)",
                   "n": int(c.sum()),
                   "purity": round(float(c.max() / c.sum()), 3),
                   f"mean_{target}": round(float(y.values[n.rows].mean()),
                                           3)}
            for lab, cc in zip(class_labels, c):
                rec[f"pct[{lab}]"] = round(100 * cc / c.sum(), 1)
            leaf_rows.append(rec)
            return
        thr_rows.append({"depth": n.depth, "split_feature": n.feature,
                         "threshold": round(n.threshold, 4),
                         "n_samples": len(n.rows)})
        f = style.short(n.feature)
        walk(n.left, conds + [f"{f} <= {n.threshold:.4g}"])
        walk(n.right, conds + [f"{f} > {n.threshold:.4g}"])

    walk(root, [])
    pd.DataFrame(thr_rows, columns=["depth", "split_feature", "threshold",
                                    "n_samples"]).to_csv(
        out_dir / f"cascade_{method}_thresholds.csv", index=False)
    pd.DataFrame(leaf_rows).to_csv(
        out_dir / f"cascade_{method}_leaves.csv", index=False)
    return [r["split_feature"] for r in thr_rows]


# =============================================================================
# Stage entry points
# =============================================================================
def run(cfg, X, y, out_dir):
    out_dir = Path(out_dir)
    c = cfg["cascade"]
    S = _figure_settings(cfg)
    _check_tree_settings(c)
    sets_path = out_dir.parent / "selection" / "selected_feature_sets.csv"
    if not sets_path.exists():
        raise FileNotFoundError(f"{sets_path} not found; run the selection "
                                f"stage first.")
    sets = pd.read_csv(sets_path, index_col=0)
    sets.index = sets.index.astype(str)
    methods = _methods(c["methods"], cfg, sets)
    if not methods:
        raise ValueError("cascade.methods is auto, but the selection stage "
                         "ran neither BorutaShap nor XGBoost; list the "
                         "methods to use in cascade.methods.")
    pools = {m: sets.index[sets[m].astype(bool)].tolist() for m in methods}
    stray = sorted({f for p in pools.values() for f in p} - set(X.columns))
    if stray:
        raise ValueError(f"The selection stage chose descriptors this "
                         f"dataset does not have: {stray}. Run the "
                         f"selection stage again.")
    edges = class_edges(c["class_edges"], y)
    _class_colors(S, len(edges) + 1)

    outputs.start(out_dir, cfg)
    unit, target = cfg["data"]["target_unit"], cfg["data"]["target"]
    codes, labels = make_classes(y, edges, unit)
    k = len(labels)
    min_leaf = max(int(c["min_leaf"]), int(c["min_leaf_frac"] * len(X)))
    counts = np.bincount(codes, minlength=k)
    print("Class counts:", dict(zip(labels, counts.tolist())))
    print(f"Adaptive: stop at purity >= {c['purity_stop']}, depth cap "
          f"{c['max_depth']}, min_leaf {min_leaf}\n")

    rule = (c["purity_stop"], c["max_depth"], cfg["models"]["random_state"])
    lines, split_feats, trees = [], {}, {}
    for method in methods:
        pool = pools[method]
        if not pool:
            print(f"{method}: selected no descriptors, so no tree")
            lines += [f"=== {method}: selected no descriptors, so no tree "
                      f"===", ""]
            continue
        Xf = X[pool].reset_index(drop=True)
        root = grow(Xf, codes, np.arange(len(Xf)), 0, k, min_leaf, rule)
        feats = export(root, y, codes, labels, out_dir, method, target)
        split_feats[method], trees[method] = feats, root
        acc_leaves = []
        leaves_in_order(root, acc_leaves)
        print(f"{method}: pool={len(pool)}, {len(acc_leaves)} leaves, "
              f"max depth {max_depth(root)}")
        for f in feats:
            print(f"    split on {style.short(f)}")
        lines.append(f"=== {method} (pool={len(pool)}, {len(acc_leaves)} "
                     f"leaves, max depth {max_depth(root)}) ===")
        lines.append("  splits (tree order): "
                     + ", ".join(style.short(f) for f in feats))
        lines.append("")

    if len(split_feats) >= 2:
        common = set.intersection(*(set(map(style.short, feats))
                                    for feats in split_feats.values()))
        which = "both" if len(split_feats) == 2 else f"all {len(split_feats)}"
        lines.append(f"Split features common to {which}: "
                     f"{sorted(common) if common else 'none'}")
    header = [
        "Classes: " + ", ".join(f"{lab} ({n} rows)"
                                for lab, n in zip(labels, counts)),
        f"A branch stops at purity >= {c['purity_stop']}, at depth "
        f"{c['max_depth']}, or below {2 * min_leaf} rows (minimum leaf "
        f"{min_leaf})"]
    (out_dir / "cascade_summary.txt").write_text(
        "\n".join(header) + "\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8")

    # what plot() needs: the trees, the rows they hold, the split descriptors
    pdata = out_dir / "plot_data"
    pdata.mkdir()
    used = list(dict.fromkeys(f for feats in split_feats.values()
                              for f in feats))
    rows = X[used].reset_index(drop=True)
    rows.insert(0, target, np.asarray(y))
    rows.to_csv(pdata / "rows.csv", index=False)
    leaf_of = {}
    for method, root in trees.items():
        nodes, leaf_of[method] = _save_tree(root, len(X))
        nodes.to_csv(pdata / f"{method}_nodes.csv", index=False)
    pd.DataFrame(leaf_of, columns=list(trees)).to_csv(
        pdata / "leaf_of_row.csv", index=False)
    (pdata / "trees.json").write_text(json.dumps(
        {"target": target, "class_edges": edges, "methods": list(trees)},
        indent=1))


def plot(cfg, out_dir):
    out_dir = Path(out_dir)
    pdata = out_dir / "plot_data"
    if not (pdata / "trees.json").exists():
        raise FileNotFoundError(f"{pdata} not found; run the cascade stage "
                                f"first.")
    S = _figure_settings(cfg)
    meta = json.loads((pdata / "trees.json").read_text())
    if not meta["methods"]:
        print("No method selected any descriptors, so there are no trees "
              "to draw.")
        return
    rows = pd.read_csv(pdata / "rows.csv")
    y = rows[meta["target"]]
    leaf_of = pd.read_csv(pdata / "leaf_of_row.csv")
    unit = cfg["data"]["target_unit"]
    codes, labels = make_classes(y, meta["class_edges"], unit)
    colors = _class_colors(S, len(labels))
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    with style.paper_style(), mpl.rc_context(style.VECTOR_RC):
        for method in meta["methods"]:
            nodes = pd.read_csv(pdata / f"{method}_nodes.csv",
                                keep_default_na=False,
                                na_values={"threshold": [""]},
                                dtype={"feature": str, "side": str})
            root = _load_tree(nodes, leaf_of[method].to_numpy())
            if root.is_leaf:
                print(f"{method}: the tree has no split (no descriptor of "
                      f"its pool separates the classes), so no figure")
                continue
            W, H, cpit, rpit = draw(
                root, rows, y, codes, labels, colors,
                str(plots_dir / f"cascade_{method}"),
                S.panel_labels.get(method), S, unit,
                cfg["data"]["target_label"])
            fitted = (cpit > S.col_inches + 1e-6
                      or rpit > S.row_inches + 1e-6)
            print(f"{method}: canvas {W:.1f}x{H:.1f} in, pitch "
                  f"{cpit:.2f}x{rpit:.2f} in"
                  + ("  [auto-fit]" if fitted else ""))
