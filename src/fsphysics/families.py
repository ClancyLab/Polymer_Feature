"""Families stage: collinear descriptors grouped into families.

Ported from feature_families_and_stability.py (Task 2) and
dendrogram_figure.py.

Spearman |rho| -> distance 1 - |rho| -> average-linkage clustering. Under
average linkage a merge height is the mean pairwise distance, so cutting at
1 - corr_threshold groups descriptors whose average |rho| is at least
corr_threshold. Families are numbered F1, F2, ... by size, then by their
first member's name. Also written: the families at compare_thresholds, the
descriptor pairs with |rho| >= corr_threshold, and a VIF table.

Family names come from labels.family_names, keyed by any member descriptor
(robust when a different cut renumbers the families) or by family id. A
family holding two named members, as can happen at a lower cut, gets both
names. Unnamed families keep their id.

run() writes the tables, with the correlation matrix twice: rounded to 4
decimals (spearman_correlation_matrix.csv, as in the original) and at full
precision (spearman_correlation_matrix_full.csv). plot() draws the figures
from them, each from the precision its original script used:
  dendrogram.png               quick right-hand dendrogram (diagnostic; full)
  corr_heatmap_clustered.png   correlations in dendrogram order (diagnostic;
                               full)
  dendrogram_topdown.png       the paper's family dendrogram (4 decimals, the
                               file dendrogram_figure.py read)
plus dendrogram_split_guide.csv, every merge of that dendrogram.

Differences from the original scripts, all deliberate:
  * The top-down dendrogram uses the same cut as the families
    (families.corr_threshold). dendrogram_figure.py had its own setting,
    and the copy sent had 0.6 against families made at 0.7.
  * The family side (Polymer/Solvent) comes from labels.blocks, so it works
    for any naming; the original split names at the first "-".
  * Branch labels may name families by a member descriptor instead of an
    id; family names come from the config instead of plot_style.py.
  * The memberships at every compare_threshold are saved
    (family_sensitivity.csv); the original printed only their counts.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from . import outputs, style


# =============================================================================
# Families
# =============================================================================
def spearman_matrix(X):
    res = spearmanr(X.values)
    rho = getattr(res, "correlation", None)
    if rho is None:
        rho = res.statistic
    rho = np.asarray(rho, dtype=float)
    if rho.ndim == 0:                         # scipy returns a scalar for 2
        rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
    rho = (rho + rho.T) / 2.0
    np.fill_diagonal(rho, 1.0)
    return pd.DataFrame(rho, index=X.columns, columns=X.columns)


def cluster_families(corr, threshold):
    """Average-linkage clustering on 1-|rho|; cut at 1-threshold."""
    dist = 1.0 - corr.abs().values
    np.fill_diagonal(dist, 0.0)
    Z = hierarchy.linkage(squareform(dist, checks=False), method="average")
    labels = hierarchy.fcluster(Z, t=1.0 - threshold, criterion="distance")
    return Z, pd.Series(labels, index=corr.index, name="cluster")


def _side(feature, blocks):
    """Block a descriptor belongs to ("Polymer-x" -> "Polymer"), or ""."""
    for prefix in blocks:
        if feature.startswith(prefix):
            return prefix.rstrip("-_: ") or prefix
    return ""


def families_table(labels, blocks):
    """Relabel clusters F1..Fn by size, add the block (side) they sit in."""
    groups = labels.groupby(labels).groups  # cluster id -> features
    ordered = sorted(groups.items(),
                     key=lambda kv: (-len(kv[1]), sorted(kv[1])[0]))
    rows = []
    for i, (_, feats) in enumerate(ordered, start=1):
        feats = sorted(feats)
        sides = {_side(f, blocks) for f in feats}
        side = sides.pop() if len(sides) == 1 else "Mixed"
        for f in feats:
            rows.append({"feature": f, "family_id": f"F{i}",
                         "family_size": len(feats), "family_side": side,
                         "family_members": "; ".join(feats)})
    return pd.DataFrame(rows).set_index("feature")


def _short_family_name(name, blocks):
    """"Polymer charge distribution" -> "P: charge distribution"."""
    for prefix, tag in blocks.items():
        word = prefix.rstrip("-_: ")
        if word and name.startswith(word + " "):
            return f"{tag}: {name[len(word) + 1:]}"
    return name


def family_names_table(fam, cfg):
    """One row per family with its name and short name."""
    names_cfg = cfg["labels"]["family_names"]
    blocks = cfg["labels"]["blocks"]
    rows = []
    for fid, grp in fam.groupby("family_id", sort=False):
        members = list(grp.index)
        if fid in names_cfg:
            names = [names_cfg[fid]]
        else:
            names = list(dict.fromkeys(names_cfg[m] for m in members
                                       if m in names_cfg))
        rows.append({
            "family_id": fid,
            "family_name": " / ".join(names) if names else fid,
            "family_short": (" / ".join(_short_family_name(n, blocks)
                                        for n in names) if names else fid),
            "family_size": len(members),
            "family_side": grp["family_side"].iloc[0],
            "family_members": grp["family_members"].iloc[0],
        })
    out = pd.DataFrame(rows)
    out["_n"] = out["family_id"].str[1:].astype(int)
    return out.sort_values("_n").drop(columns="_n").set_index("family_id")


def high_corr_pairs(corr, threshold):
    rows = []
    cols = corr.columns
    for i, j in combinations(range(len(cols)), 2):
        r = corr.iloc[i, j]
        if abs(r) >= threshold:
            rows.append({"feature_1": cols[i], "feature_2": cols[j],
                         "spearman_rho": round(float(r), 4)})
    return (pd.DataFrame(rows)
            .sort_values("spearman_rho", key=lambda s: s.abs(),
                         ascending=False)
            if rows else pd.DataFrame(
                columns=["feature_1", "feature_2", "spearman_rho"]))


def vif_table(X):
    Xs = StandardScaler().fit_transform(X)
    rows = []
    for j, feat in enumerate(X.columns):
        others = [k for k in range(X.shape[1]) if k != j]
        r2 = LinearRegression().fit(Xs[:, others], Xs[:, j]).score(
            Xs[:, others], Xs[:, j])
        vif = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
        rows.append({"feature": feat, "R2_vs_rest": round(float(r2), 4),
                     "VIF": round(float(vif), 2)})
    return pd.DataFrame(rows).set_index("feature").sort_values(
        "VIF", ascending=False)


# =============================================================================
# Diagnostic figures (matplotlib default style, as in the original)
# =============================================================================
def plot_dendrogram(Zlink, feats, threshold, out_path):
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(feats))))
    hierarchy.dendrogram(Zlink, labels=list(feats), orientation="right",
                         color_threshold=1.0 - threshold, ax=ax)
    ax.axvline(1.0 - threshold, color="red", ls="--", lw=1,
               label=f"cut: mean |\u03c1| = {threshold}")
    ax.set_xlabel("average-linkage distance (1 - |Spearman \u03c1|)")
    ax.set_title("Feature families (hierarchical clustering)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return hierarchy.leaves_list(Zlink)


def plot_corr_heatmap(corr, leaf_order, out_path):
    order = [corr.columns[i] for i in leaf_order]
    c = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6)
    for i in range(len(order)):
        for j in range(len(order)):
            v = c.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4,
                    color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Spearman \u03c1")
    ax.set_title("Spearman correlation (dendrogram order)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =============================================================================
# Top-down dendrogram (dendrogram_figure.py)
# =============================================================================
def leaf_sets(Z, n):
    """{node index -> set of leaf indices under it}, leaves included."""
    sets = {i: {i} for i in range(n)}
    for i, (l, r, _, _) in enumerate(Z):
        sets[n + i] = sets[int(l)] | sets[int(r)]
    return sets


def node_coords(feats, Z):
    """Coordinates scipy uses when drawing: leaves sit at 5, 15, 25, ... in
    leaf order; an internal node sits at the midpoint of its two children,
    at the height of its merge distance."""
    n = len(feats)
    x = {leaf: 5.0 + 10.0 * rank
         for rank, leaf in enumerate(hierarchy.leaves_list(Z))}
    y = {i: 0.0 for i in range(n)}
    for i, (l, r, h, _) in enumerate(Z):
        x[n + i] = 0.5 * (x[int(l)] + x[int(r)])
        y[n + i] = float(h)
    return x, y


def parents(Z, n):
    p = {}
    for i, (l, r, _, _) in enumerate(Z):
        p[int(l)] = p[int(r)] = n + i
    return p


def build_colors(feats, Z, fam_id, cfg):
    """Family colours from a ramp, ordered by position along the dendrogram,
    so adjacent (more similar) families get adjacent colours."""
    d = cfg["families"]["dendrogram"]
    n = len(feats)
    order = hierarchy.leaves_list(Z)                 # left-to-right leaf order
    pos = {feats[leaf]: i for i, leaf in enumerate(order)}

    fam_pos = {}
    for f in fam_id.unique():
        members = [x for x in feats if fam_id[x] == f]
        fam_pos[f] = np.mean([pos[m] for m in members])

    scheme = d["color_scheme"]
    if scheme not in ("single", "blocks"):
        raise ValueError(f"families.dendrogram.color_scheme: {scheme!r} "
                         f"(use single or blocks)")
    if scheme == "blocks" and not cfg["labels"]["blocks"]:
        print("  note: color_scheme 'blocks' needs labels.blocks; using the "
              "single colour ramp")
        scheme = "single"

    fam_color = {}
    if scheme == "single":
        lo, hi = d["cmap_range"]
        cmap = plt.get_cmap(d["cmap"])
        fams = sorted(fam_pos, key=lambda f: fam_pos[f])
        for i, f in enumerate(fams):
            t = lo + (hi - lo) * (i / max(len(fams) - 1, 1))
            fam_color[f] = matplotlib.colors.to_hex(cmap(t))
    else:
        # one ramp per block (labels.blocks order), ramped within each block
        blocks = list(cfg["labels"]["blocks"])
        for b, prefix in enumerate(blocks):
            cname, lo, hi = d["block_cmaps"][b % len(d["block_cmaps"])]
            side_fams = sorted(
                [f for f in fam_pos
                 if all(x.startswith(prefix)
                        for x in feats if fam_id[x] == f)],
                key=lambda f: fam_pos[f])
            cmap = plt.get_cmap(cname)
            for i, f in enumerate(side_fams):
                t = lo + (hi - lo) * (i / max(len(side_fams) - 1, 1))
                fam_color[f] = matplotlib.colors.to_hex(cmap(t))
        mixed = [f for f in fam_pos if f not in fam_color]
        if mixed:
            print(f"  note: families {mixed} span more than one block and are "
                  f"drawn in the above-cut grey; color_scheme 'single' gives "
                  f"every family its own colour")
        for f in fam_pos:                             # safety net
            fam_color.setdefault(f, d["above_cut_color"])

    sets = leaf_sets(Z, n)

    def link_color_func(k):
        fams = {fam_id[feats[i]] for i in sets[k]}
        if len(fams) == 1:
            return fam_color[fams.pop()]
        return d["above_cut_color"]

    return fam_color, link_color_func


def annotate_branches(ax, feats, Z, fam_id, cfg):
    """Place each branch label on the smallest node covering its families
    (given as family ids or member descriptors). Returns what was drawn."""
    d = cfg["families"]["dendrogram"]
    weight = "bold" if d["bold"] else "normal"
    n = len(feats)
    sets, (xs, ys), par = leaf_sets(Z, n), node_coords(feats, Z), parents(Z, n)
    present = set(fam_id.unique())
    top = float(Z[-1, 2])
    drawn = []

    for spec in d["branch_labels"]:
        text = spec["text"]
        fams, unknown = set(), []
        for ref in spec["families"]:
            if ref in present:
                fams.add(ref)
            elif ref in fam_id.index:
                fams.add(fam_id[ref])
            else:
                unknown.append(ref)
        if unknown:
            print(f"  label {text!r}: skipped, {unknown} is not a family or "
                  f"descriptor in this run")
            continue

        targets = {i for i, f in enumerate(feats) if fam_id[f] in fams}
        node = min((k for k in range(n, n + len(Z)) if targets <= sets[k]),
                   key=lambda k: len(sets[k]), default=None)
        if node is None:
            print(f"  label {text!r}: skipped, no node covers {sorted(fams)}")
            continue

        h = ys[node]
        h_up = ys[par[node]] if node in par else top * 1.08
        h_dn = max(ys[int(Z[node - n, 0])], ys[int(Z[node - n, 1])])
        gap_up, gap_dn = h_up - h, h - h_dn
        y = (h + d["label_gap_frac"] * gap_up
             if gap_up >= d["min_gap_above"] * top      # room above the bar
             else h - d["label_gap_frac"] * gap_dn)     # otherwise under it
        y += spec.get("dy", 0.0)

        ax.text(xs[node] + spec.get("dx", 0.0), y, text,
                ha="center", va="center", zorder=6,
                color=d["branch_label_color"], fontweight=weight,
                linespacing=1.15,
                fontsize=(d["block_label_fontsize"]
                          if spec.get("kind") == "block"
                          else d["branch_label_fontsize"]),
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="square,pad=0.18", alpha=0.9))
        drawn.append((text.replace("\n", " "), h, len(sets[node]), y))

    return drawn


def split_guide(feats, Z, fam_id, threshold, out_csv):
    """Every merge of the dendrogram, for placing branch labels."""
    n = len(feats)

    def members(i):
        if i < n:
            return [feats[i]]
        return members(int(Z[i - n, 0])) + members(int(Z[i - n, 1]))

    rows = []
    for i, (l, r, h, _) in enumerate(Z):
        ml, mr = members(int(l)), members(int(r))
        rows.append({
            "merge": i + 1,
            "height": round(float(h), 4),
            "mean_abs_rho": round(1.0 - float(h), 3),
            "above_cut": bool(h > 1.0 - threshold),
            "n_left": len(ml), "n_right": len(mr),
            "left_families": ",".join(sorted({fam_id[x] for x in ml})),
            "right_families": ",".join(sorted({fam_id[x] for x in mr})),
            "left_members": "; ".join(style.short(x) for x in ml),
            "right_members": "; ".join(style.short(x) for x in mr),
        })
    df = pd.DataFrame(rows).sort_values("height", ascending=False)
    df.to_csv(out_csv, index=False)
    return df


def plot_topdown(feats, Z, fam_id, fam_color, link_color_func, names,
                 threshold, cfg, out_path):
    d = cfg["families"]["dendrogram"]
    weight = "bold" if d["bold"] else "normal"
    cut_height = 1.0 - threshold

    fig, ax = plt.subplots(figsize=tuple(d["figsize"]))
    hierarchy.dendrogram(
        Z, labels=[style.short(f) for f in feats], orientation="top",
        link_color_func=link_color_func, ax=ax,
        above_threshold_color=d["above_cut_color"])

    ax.axhline(cut_height, color=d["cut_color"], ls="--", lw=2.0, zorder=0)
    ax.text(0.995, cut_height, f" cut: mean |\u03c1| = {threshold} ",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            color=d["cut_color"], fontsize=12, fontweight=weight)

    ax.set_ylabel("Average-linkage distance  (1 \u2212 |Spearman \u03c1|)",
                  fontsize=14, fontweight=weight)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=d["label_fontsize"], colors="black")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontweight=weight)

    # each tick label in its family colour, outlined in black so the pale
    # end of the ramp stays legible
    for lab in ax.get_xticklabels():
        full = next((f for f in feats if style.short(f) == lab.get_text()),
                    None)
        if full is not None:
            lab.set_color(fam_color[fam_id[full]])
        if d["label_outline"]:
            lab.set_path_effects([pe.withStroke(linewidth=d["label_outline"],
                                                foreground="black")])

    print("Branch labels:")
    for text, h, k, y in annotate_branches(ax, feats, Z, fam_id, cfg):
        print(f"  {text:<20s} node h={h:.3f} ({k:2d} descriptors)"
              f"  ->  drawn at y={y:.3f}")

    # secondary axis: distance -> mean |rho|
    sec = ax.secondary_yaxis(
        "right", functions=(lambda v: 1 - v, lambda r: 1 - r))
    sec.set_ylabel("mean |Spearman \u03c1|", fontsize=13, fontweight=weight)
    sec.tick_params(labelsize=11)

    for s in ("top", "right"):
        ax.spines[s].set_visible(True)

    fams_sorted = sorted(fam_color, key=lambda f: int(f[1:]))
    handles = [Line2D([0], [0], color=fam_color[f], lw=4,
                      label=names.get(f, f))
               for f in fams_sorted]
    handles.append(Line2D([0], [0], color=d["above_cut_color"], lw=4,
                          label="above cut (between families)"))
    ax.legend(handles=handles, fontsize=10, frameon=False,
              loc="upper left", bbox_to_anchor=(1.06, 1.0),
              title="Feature family", title_fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Stage entry points
# =============================================================================
def run(cfg, X, y, out_dir):
    out_dir = str(out_dir)
    p = lambda name: os.path.join(out_dir, name)        # noqa: E731
    f = cfg["families"]
    blocks = cfg["labels"]["blocks"]
    thr = f["corr_threshold"]
    if X.shape[1] < 2:
        raise ValueError("Families need at least two descriptors.")
    outputs.start(out_dir, cfg)
    summary = []

    corr = spearman_matrix(X)
    corr.round(4).to_csv(p("spearman_correlation_matrix.csv"))
    corr.to_csv(p("spearman_correlation_matrix_full.csv"))

    _, labels = cluster_families(corr, thr)
    fam = families_table(labels, blocks)
    fam.to_csv(p("feature_families.csv"))
    names = family_names_table(fam, cfg)
    names.to_csv(p("family_names.csv"))
    n_fam = fam["family_id"].nunique()
    print(f"{n_fam} families at mean |rho| >= {thr}")
    summary.append(f"Families at |rho|>={thr}: {n_fam}")
    known = set(X.columns) | set(fam["family_id"])
    for key in cfg["labels"]["family_names"]:
        if key not in known:
            print(f"  note: labels.family_names key '{key}' is not a "
                  f"descriptor or family id in this run")

    sens = pd.DataFrame(index=fam.index)
    for t in f["compare_thresholds"]:
        _, lab = cluster_families(corr, t)
        sens[f"family_{t}"] = families_table(lab, blocks)["family_id"]
        print(f"  threshold {t} -> {lab.nunique()} families")
        summary.append(f"  threshold {t}: {lab.nunique()} families")
    sens.to_csv(p("family_sensitivity.csv"))

    pairs = high_corr_pairs(corr, thr)
    pairs.to_csv(p("high_correlation_pairs.csv"), index=False)
    print(f"  {len(pairs)} pairs with |rho| >= {thr}")
    summary.append(f"Pairs with |rho| >= {thr}: {len(pairs)}")

    vif = vif_table(X)
    vif.to_csv(p("vif_table.csv"))
    n_high = int((vif["VIF"] > 5).sum())
    vif_max = vif["VIF"].replace(np.inf, np.nan).max()
    print(f"  VIF > 5 for {n_high} descriptors (max = {vif_max:.1f})")
    summary.append(f"VIF > 5: {n_high} descriptors (max {vif_max:.1f})")

    with open(p("families_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary) + "\n")


def _linkage(corr):
    D = 1.0 - corr.abs().values
    np.fill_diagonal(D, 0.0)
    return hierarchy.linkage(squareform(D, checks=False), method="average")


def plot(cfg, out_dir):
    out_dir = str(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    thr = cfg["families"]["corr_threshold"]

    corr = pd.read_csv(os.path.join(out_dir,
                                    "spearman_correlation_matrix.csv"),
                       index_col=0)
    fam = pd.read_csv(os.path.join(out_dir, "feature_families.csv"),
                      index_col=0)
    names = pd.read_csv(os.path.join(out_dir, "family_names.csv"),
                        index_col=0)["family_name"].to_dict()
    feats = list(corr.columns)
    fam_id = fam["family_id"].reindex(feats)
    Z = _linkage(corr)

    # the diagnostic figures from the full-precision matrix, as in the
    # original (folders made before that file existed use the rounded one)
    full_path = os.path.join(out_dir, "spearman_correlation_matrix_full.csv")
    corr_full = (pd.read_csv(full_path, index_col=0)
                 if os.path.exists(full_path) else corr)
    with style.default_style():
        leaf_order = plot_dendrogram(_linkage(corr_full), corr_full.columns,
                                     thr,
                                     os.path.join(plots_dir, "dendrogram.png"))
        plot_corr_heatmap(corr_full, leaf_order,
                          os.path.join(plots_dir, "corr_heatmap_clustered.png"))

    with style.paper_style():
        fam_color, lcf = build_colors(feats, Z, fam_id, cfg)
        plot_topdown(feats, Z, fam_id, fam_color, lcf, names, thr, cfg,
                     os.path.join(plots_dir, "dendrogram_topdown.png"))
    guide = split_guide(feats, Z, fam_id, thr,
                        os.path.join(out_dir, "dendrogram_split_guide.csv"))
    print("Splits above the cut (top-down), for placing branch labels:")
    for _, r in guide[guide.above_cut].iterrows():
        print(f"  h={r.height:.3f} (|rho|={r.mean_abs_rho:.2f})  "
              f"[{r.left_families}] {r.n_left} vs [{r.right_families}] "
              f"{r.n_right}")
