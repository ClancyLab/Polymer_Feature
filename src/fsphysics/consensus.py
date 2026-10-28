"""Consensus stage: how far the selectors agree, per descriptor and per family.

Ported from feature_families_and_stability.py (Task 3a). From the selection
stage's unified-SHAP importances:
  * pairwise Spearman and Kendall tau between the methods' importance
    vectors, and Kendall's W concordance with its chi-square p,
  * top-k Jaccard overlaps for each k in consensus.top_ks,
  * Borda (mean-rank) consensus with the spread across methods,
  * the same statistics at FAMILY level (importance summed over each
    family's members). If the methods disagree only on which collinear
    descriptor gets the credit, family-level agreement is much higher than
    descriptor-level agreement: this is the falsifiable check.

Reads selection/feature_importance_shap_comparison.csv and
families/feature_families.csv from the same run. run() writes the tables;
plot() draws consensus_ranks.png from consensus_rank_table.csv.

Differences from the original, all deliberate: the statistics go to
consensus_summary.txt (the original wrote them into stability_summary.txt,
which now belongs to the stability stage), and any two or more selection
methods are compared, not a fixed three.
"""

import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, kendalltau, spearmanr

from . import outputs, style


def kendalls_w(rank_df):
    """Kendall's coefficient of concordance across columns (rankers)."""
    m = rank_df.shape[1]
    n = rank_df.shape[0]
    R = rank_df.sum(axis=1)
    S = float(((R - R.mean()) ** 2).sum())
    W = 12.0 * S / (m * m * (n ** 3 - n))
    chi2_stat = m * (n - 1) * W
    p = float(chi2.sf(chi2_stat, n - 1))
    return W, chi2_stat, p


def consensus_stats(imp, label, out_lines, top_ks):
    """Pairwise rank correlations, Kendall's W, top-k Jaccard, Borda table."""
    ranks = imp.rank(ascending=False)  # average ranks (ties unlikely)

    pair_rows = []
    for a, b in combinations(imp.columns, 2):
        rho, _ = spearmanr(imp[a], imp[b])
        tau, _ = kendalltau(imp[a], imp[b])
        pair_rows.append({"level": label, "method_a": a, "method_b": b,
                          "spearman": round(float(rho), 3),
                          "kendall_tau": round(float(tau), 3)})
    pairs = pd.DataFrame(pair_rows)

    W, c2, p = kendalls_w(ranks)
    out_lines.append(f"[{label}] Kendall W = {W:.3f} "
                     f"(chi2 = {c2:.1f}, df = {len(imp)-1}, p = {p:.2e})")
    for r in pair_rows:
        out_lines.append(f"[{label}]   {r['method_a']:>10s} vs "
                         f"{r['method_b']:<10s} spearman = {r['spearman']:.3f}"
                         f"  kendall = {r['kendall_tau']:.3f}")

    jac_rows = []
    for k in top_ks:
        k_eff = min(k, len(imp))
        tops = {c: set(imp[c].sort_values(ascending=False).head(k_eff).index)
                for c in imp.columns}
        inter = set.intersection(*tops.values())
        union = set.union(*tops.values())
        row = {"level": label, "k": k_eff,
               "jaccard_3way": round(len(inter) / len(union), 3),
               "n_shared_all": len(inter)}
        for a, b in combinations(imp.columns, 2):
            row[f"jaccard_{a}_{b}"] = round(
                len(tops[a] & tops[b]) / len(tops[a] | tops[b]), 3)
        jac_rows.append(row)
        out_lines.append(f"[{label}] top-{k_eff}: 3-way Jaccard = "
                         f"{row['jaccard_3way']:.2f} (shared by all: "
                         f"{len(inter)})")
    jac = pd.DataFrame(jac_rows)

    borda = pd.DataFrame({
        **{f"{m}_rank": ranks[m].round(1) for m in imp.columns},
        "consensus_mean_rank": ranks.mean(axis=1).round(2),
        "rank_spread": (ranks.max(axis=1) - ranks.min(axis=1)).round(1),
        "rank_std": ranks.std(axis=1).round(2),
    }).sort_values("consensus_mean_rank")

    return pairs, jac, borda, W


def plot_consensus(borda, out_path):
    """Descriptors on the X axis; rank on Y (rank 1 = most important, top)."""
    b = borda.sort_values("consensus_mean_rank")        # best first (left)
    rank_cols = [c for c in b.columns if c.endswith("_rank")
                 and c != "rank_spread"]
    lo = b[rank_cols].min(axis=1)
    hi = b[rank_cols].max(axis=1)
    x = np.arange(len(b))

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(b) + 2.5), 6.2))
    ax.errorbar(x, b["consensus_mean_rank"],
                yerr=[b["consensus_mean_rank"] - lo,
                      hi - b["consensus_mean_rank"]],
                fmt="o", ms=7, color="#0072B2", markeredgecolor="black",
                markeredgewidth=0.8, ecolor="black", elinewidth=1.3,
                capsize=4, capthick=1.3, linestyle="none")
    ax.set_xticks(x)
    ax.set_xticklabels([style.short(f) for f in b.index])
    style.rotate_xticklabels(ax, fontsize=11)
    ax.set_ylabel("Rank across methods", fontsize=13)
    ax.invert_yaxis()                       # rank 1 at the top
    ax.grid(False)
    ax.set_xlim(-0.7, len(b) - 0.3)
    style.frame_axes(ax, linewidth=1.3)
    ax.tick_params(axis="y", labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(cfg, X, y, out_dir):
    out_dir = str(out_dir)
    p = lambda name: os.path.join(out_dir, name)        # noqa: E731
    run_dir = Path(out_dir).parent
    shap_csv = run_dir / "selection" / "feature_importance_shap_comparison.csv"
    fam_csv = run_dir / "families" / "feature_families.csv"
    for path, stage in ((shap_csv, "selection"), (fam_csv, "families")):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run the {stage} "
                                    f"stage first.")
    top_ks = cfg["consensus"]["top_ks"]

    shap_tbl = pd.read_csv(shap_csv, index_col=0)
    methods = [m for m in cfg["selection"]["methods"]
               if f"{m}_importance" in shap_tbl]
    if len(methods) < 2:
        raise ValueError("Consensus needs at least two selection methods.")
    outputs.start(out_dir, cfg)
    imp = shap_tbl[[m + "_importance" for m in methods]].copy()
    imp.columns = methods
    imp = imp.reindex(X.columns)
    fam = pd.read_csv(fam_csv, index_col=0)
    summary = []

    pairs_feat, jac_feat, borda_feat, W_feat = consensus_stats(
        imp, "feature", summary, top_ks)
    fam_imp = imp.copy()
    fam_imp["family_id"] = fam["family_id"].reindex(fam_imp.index)
    fam_sum = fam_imp.groupby("family_id")[methods].sum()
    pairs_fam, jac_fam, borda_fam, W_fam = consensus_stats(
        fam_sum, "family", summary, top_ks)
    print(f"Kendall W: descriptor level = {W_feat:.3f}, "
          f"family level = {W_fam:.3f}")

    pd.concat([pairs_feat, pairs_fam], ignore_index=True).to_csv(
        p("method_rank_correlations.csv"), index=False)
    pd.concat([jac_feat, jac_fam], ignore_index=True).to_csv(
        p("topk_jaccard.csv"), index=False)
    borda_feat.to_csv(p("consensus_rank_table.csv"))

    fam_out = fam_sum.copy()
    fam_out["members"] = fam.groupby("family_id")["family_members"].first()
    fam_out["consensus_mean_rank"] = borda_fam["consensus_mean_rank"]
    fam_out.sort_values("consensus_mean_rank").to_csv(
        p("family_importance.csv"))

    with open(p("consensus_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary) + "\n")
    for line in summary:
        print(f"  {line}")


def plot(cfg, out_dir):
    out_dir = str(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    borda = pd.read_csv(os.path.join(out_dir, "consensus_rank_table.csv"),
                        index_col=0)
    with style.default_style():
        plot_consensus(borda, os.path.join(plots_dir, "consensus_ranks.png"))
