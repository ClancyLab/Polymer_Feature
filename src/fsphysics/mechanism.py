"""Mechanism stage: why the top descriptors matter for the target.

Ported from mechanism_analysis.py. Five parts, run in this order; each can be
switched on or off in mechanism.parts:

  interactions           XGBoost SHAP interaction values: the full
                         descriptor matrix, the top pairs, each descriptor's
                         strongest partner, the share of interaction strength
                         between and within blocks (polymer x solvent in the
                         paper), and the family-level aggregation, reported
                         as two figures: main effects (the diagonal) and
                         cross-family interactions (one off-diagonal
                         triangle), so neither colour scale is set by the
                         other.
  dependence_ale         per key descriptor: XGBoost SHAP dependence,
                         coloured by its strongest partner, beside 1-D
                         accumulated local effects (Apley and Zhu) for
                         XGBoost and the RandomForest. ALE rather than
                         partial dependence, because it does not extrapolate
                         into combinations of correlated descriptors that
                         never occur.
  dependence_comparison  SHAP dependence panels, one file per model (XGBoost,
                         RandomForest), for the key descriptors coloured by
                         their strongest partner, and for chosen pairs.
  lofo                   leave-one-family-out: drop each family, refit the
                         downstream RandomForest with K-fold CV on the full
                         dataset, and report the change in R2, RMSE and MAE.
                         Families correlate below the clustering cut, so the
                         remaining ones partly compensate: the changes are
                         lower bounds on a family's unique contribution.
  surrogates (off by     shallow decision trees on the full dataset, giving
  default)               explicit thresholds: one depth-2 tree per key
                         descriptor and one depth-3 tree on a few of them.
                         Drawing the trees needs dtreeviz and Graphviz; the
                         thresholds and rules do not.

The explanation models (the selection stage's XGBoost, and its BorutaShap
RandomForest settings) are fitted on the selection split, so every
attribution here matches the reported importances. LOFO and the surrogate
trees use the full dataset.

run() starts from an empty stage folder, fits, and writes the tables plus,
in plot_data/, the full-precision values the figures are drawn from. plot()
draws every figure from those, except the surrogate tree drawings
(surrogate_trees/), which run() makes because they need the fitted trees.

Differences from mechanism_analysis.py, all deliberate:
  * The surrogate trees are off by default (the script ran them); they are
    no longer in the paper. Add "surrogates" to mechanism.parts for them.
  * Descriptor labels come from labels.short_names and labels.blocks, as in
    every other stage ("P: quadrupole"); the script's own short() gave
    "P:quadrupole" in the feature-level heatmap, the dependence+ALE panels,
    the summary and the tables.
  * labels.blocks replaces the hard-coded Polymer/Solvent split: the pair
    types (P-S, P-P, S-S in the paper), their shares and the grouped family
    figure work for any blocks, and are left out for a dataset without them.
    Pairs with a descriptor in no block are their own type, "other", so the
    shares always add up to 100%. The block outline on the grouped figure is
    also right for the upper triangle, which the script outlined in the
    empty half.
  * Family names come from the families stage. Figures take descriptor and
    family labels from the current settings, so `fsphysics plot` relabels
    them without refitting; the tables record the labels of the run.
  * The key descriptors, the pair plots, the partner report (hardness in the
    paper) and the joint-tree descriptors are settings. By default they are
    chosen from the data: the top descriptors by consensus rank, and the
    strongest interaction pairs. Names that are not descriptors are an
    error rather than being skipped. The partner report's summary heading
    names its descriptor ("Strongest interaction partners of P: hardness:"
    rather than "Hardness: strongest interaction partners:").
  * Units follow data.target_unit; the script assumed eV.
  * LOFO's baseline row gives the real descriptor count (the script wrote
    "all 21" for any dataset).
  * A descriptor whose SHAP values do not vary gets effect direction "none",
    and an exact zero correlation "0"; the script called both "-".
  * RandomForest predictions (LOFO, ALE) run on one thread so results repeat
    exactly. The XGBoost SHAP values are computed once for both dependence
    parts (the script computed the same values twice).
  * With the interactions part off, the strongest partners are still found
    the same way and colour both dependence parts (the script left the
    dependence+ALE panels uncoloured then).
  * mechanism_summary.txt is rewritten on every run (the script appended).
    New tables: interaction_partners.csv (each descriptor's strongest
    partner) and interaction_pair_types.csv (the block shares).
  * Layout: tables in mechanism/, figures in mechanism/plots/ (with
    dependence_ale/ and dependence_comparison/), surrogate tree drawings in
    mechanism/surrogate_trees/, the comparison's directions as
    effect_directions_by_model.csv, and file names from the descriptor
    names, with characters Windows does not allow replaced.
  * dtreeviz and Graphviz are needed only to draw surrogate trees, and are
    checked before anything is fitted; the script required them to start.
"""

import json
import os
import shutil
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, export_text

from . import models, outputs, style
from .data import selection_split
from .families import family_names_table

USES_MODELS = True         # runs with the tuned models when tuning applies

PARTS = ("interactions", "dependence_ale", "dependence_comparison", "lofo",
         "surrogates")
PARTNER_CMAP = "coolwarm"            # plot_style.PARTNER_CMAP


# =============================================================================
# Settings, units, names
# =============================================================================
def _parts(cfg):
    """The parts to run, in run order, after checking the figure settings."""
    m = cfg["mechanism"]
    parts = list(m["parts"])
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise KeyError(f"mechanism.parts: unknown {unknown}; choose from "
                       f"{list(PARTS)}")
    bad = [o for o in m["interaction_orderings"]
           if o not in ("numeric", "grouped")]
    if bad:
        raise ValueError(f"mechanism.interaction_orderings: {bad} (use "
                         f"numeric or grouped)")
    if m["interaction_triangle"] not in ("lower", "upper"):
        raise ValueError(f"mechanism.interaction_triangle: "
                         f"{m['interaction_triangle']!r} (use lower or upper)")
    return [p for p in PARTS if p in parts]


def _units(cfg):
    """(target unit, family-figure scale, family-figure unit)."""
    unit = cfg["data"]["target_unit"]
    choice = cfg["mechanism"]["family_unit"]
    if choice == "auto":
        choice = "meV" if unit == "eV" else unit
    if choice == "meV" and unit == "eV":
        return unit, 1000.0, "meV"
    if choice == unit:
        return unit, 1.0, unit
    raise ValueError(f"mechanism.family_unit {choice!r} does not fit the "
                     f"target unit {unit!r}: use auto, the target unit, or "
                     f"meV for an eV target.")


def _with_unit(text, unit):
    return f"{text} ({unit})" if unit else text


def _num(value, fmt, unit):
    return f"{value:{fmt}} {unit}" if unit else f"{value:{fmt}}"


def _suffix(unit):
    return f"_{unit}" if unit else ""


def _check_names(names, feats, setting):
    missing = [n for n in names if n not in feats]
    if missing:
        raise KeyError(f"mechanism.{setting}: {missing} are not descriptors "
                       f"of this dataset.")


def _block_tag(feature, blocks):
    for prefix, tag in blocks.items():
        if feature.startswith(prefix):
            return tag
    return None


def _pair_type(a, b, blocks):
    """"P-S" for a polymer-solvent pair, "P-P", "S-S" (tags in block order);
    "other" when either descriptor is in no block."""
    tags = list(dict.fromkeys(blocks.values()))
    ta, tb = _block_tag(a, blocks), _block_tag(b, blocks)
    if ta is None or tb is None:
        return "other"
    ta, tb = sorted((ta, tb), key=tags.index)
    return f"{ta}-{tb}"


def _pair_type_order(blocks):
    """Between-block types first, then within-block: P-S, P-P, S-S."""
    tags = list(dict.fromkeys(blocks.values()))
    between = [f"{a}-{b}" for i, a in enumerate(tags) for b in tags[i + 1:]]
    return between + [f"{a}-{a}" for a in tags]


def _read_families(path):
    fam = pd.read_csv(path, index_col=0)
    fam["family_side"] = fam["family_side"].fillna("").astype(str)
    return fam


def _family_info(fam, cfg):
    """Per family id: figure label, LOFO label, summary text and side, from
    the membership the stage used and the current settings."""
    names = family_names_table(fam, cfg)
    info = {}
    for fid, row in names.iterrows():
        short = row["family_short"]
        named = short != fid
        info[fid] = {
            "label": (short if cfg["mechanism"]["family_fig_names"] and named
                      else fid),
            "short": short,
            "text": f"{fid} ({short})" if named else fid,
            "side": row["family_side"],
        }
    return info


def _family_groups(fam_ids, info, blocks):
    """Families by block (labels.blocks order); any others last."""
    sides = [prefix.rstrip("-_: ") or prefix for prefix in blocks]
    groups = [[f for f in fam_ids if info[f]["side"] == side]
              for side in sides]
    groups.append([f for f in fam_ids if info[f]["side"] not in sides])
    return [g for g in groups if g]


def _sorted_family_ids(fid):
    return sorted(fid.unique(), key=lambda s: int(s[1:]))


def _save_pdata(df, path, index=True):
    """Save plot data so it reads back exactly. XGBoost's SHAP values are
    32-bit: such columns are written as the 64-bit numbers they equal, and
    plot_data/dtypes.json records them so they are restored as 32-bit."""
    f32 = [str(c) for c in df.columns if df[c].dtype == np.float32]
    df.astype({c: np.float64 for c in df.columns
               if df[c].dtype == np.float32}).to_csv(path, index=index)
    manifest = path.parent / "dtypes.json"
    record = json.loads(manifest.read_text()) if manifest.exists() else {}
    record[path.name] = f32
    manifest.write_text(json.dumps(record, indent=1))


def _read_pdata(pdata, name, **kwargs):
    path = pdata / name
    if not path.exists():
        return None
    df = pd.read_csv(path, **kwargs)
    manifest = pdata / "dtypes.json"
    f32 = (json.loads(manifest.read_text()).get(name, [])
           if manifest.exists() else [])
    return df.astype({c: np.float32 for c in f32}) if f32 else df


# =============================================================================
# Interactions
# =============================================================================
def interaction_matrix(xgb, X_sample):
    """Mean |SHAP interaction| over the rows: (d, d), diagonal = main
    effects."""
    inter = shap.TreeExplainer(xgb).shap_interaction_values(X_sample)
    inter = np.abs(np.asarray(inter))
    while inter.ndim > 3:
        inter = inter.mean(axis=-1)
    return inter.mean(axis=0)


def strongest_partners(M, feats):
    """Partner of each descriptor: argmax of M[i,j] + M[j,i], diagonal
    excluded. Returns (partners, pair-strength matrix)."""
    P = M + M.T
    np.fill_diagonal(P, -np.inf)
    return {feats[i]: feats[int(np.argmax(P[i]))]
            for i in range(len(feats))}, P


def pair_table(M, feats, blocks, unit):
    """Every descriptor pair with its strength (SHAP splits a pair's
    interaction across both cells), strongest first."""
    u = _suffix(unit)
    rows = []
    for i, j in combinations(range(len(feats)), 2):
        a, b = feats[i], feats[j]
        row = {"feature_1": a, "feature_2": b}
        if blocks:
            row["pair_type"] = _pair_type(a, b, blocks)
        row[f"strength{u}"] = float(M[i, j] + M[j, i])
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values(f"strength{u}", ascending=False)
            .reset_index(drop=True))


def interactions(M, feats, fam, info, cfg, out_dir, pdata, summary):
    """Interaction tables and summary (figures: plot())."""
    unit, scale, funit = _units(cfg)
    u = _suffix(unit)
    dec = 1 if scale == 1000.0 else 4
    blocks = cfg["labels"]["blocks"]
    short = style.short

    Mdf = pd.DataFrame(M, index=feats, columns=feats)
    Mdf.round(6).to_csv(out_dir / "interaction_matrix.csv")
    _save_pdata(Mdf, pdata / "interaction_matrix.csv")

    pairs = pair_table(M, feats, blocks, unit)
    pairs.round(5).to_csv(out_dir / "interaction_top_pairs.csv", index=False)

    if blocks:
        cross = pairs.groupby("pair_type")[f"strength{u}"].sum()
        total = float(cross.sum())
        summary.append("Interaction strength by pair type "
                       "(share of total off-diagonal):")
        types = _pair_type_order(blocks)
        if "other" in cross.index:
            types.append("other")
        type_rows = []
        for t in types:
            v = float(cross.get(t, 0.0))
            share = 100 * v / total if total else float("nan")
            note = ("  (pairs with a descriptor in no block)"
                    if t == "other" else "")
            summary.append(f"  {t}: {_num(v, '.4f', unit)}  ({share:.1f}%)"
                           f"{note}")
            type_rows.append({"pair_type": t, f"strength{u}": round(v, 5),
                              "share_pct": round(share, 3)})
        pd.DataFrame(type_rows).to_csv(out_dir / "interaction_pair_types.csv",
                                       index=False)
    summary.append("Top 5 interacting pairs:")
    for _, r in pairs.head(5).iterrows():
        kind = f" [{r['pair_type']}]" if blocks else ""
        summary.append(f"  {short(r['feature_1'])} x {short(r['feature_2'])}"
                       f"{kind} = {_num(r[f'strength{u}'], '.4f', unit)}")

    partner, P = strongest_partners(M, feats)
    pd.DataFrame({
        "strongest_partner": [partner[f] for f in feats],
        f"pair_strength{u}": [float(P[i].max()) for i in range(len(feats))],
    }, index=pd.Index(feats, name="feature")).round(6).to_csv(
        out_dir / "interaction_partners.csv")
    for name in cfg["mechanism"]["partner_report"]:
        row = pd.Series(P[feats.index(name)], index=feats).sort_values(
            ascending=False)
        summary.append(f"Strongest interaction partners of {short(name)}:")
        for f, v in row.head(3).items():
            summary.append(f"  {short(f)} = {_num(v, '.4f', unit)}")

    # family-level aggregation
    fid = fam["family_id"]
    fam_ids = _sorted_family_ids(fid)
    F = pd.DataFrame(0.0, index=fam_ids, columns=fam_ids)
    for i, a in enumerate(feats):
        for j, b in enumerate(feats):
            F.loc[fid[a], fid[b]] += M[i, j]
    F.round(5).to_csv(out_dir / "interaction_family_matrix.csv")
    _save_pdata(F, pdata / "interaction_family_matrix.csv")

    # the three terms the two figures report:
    #   main      = sum of M[i,i] over the family's members
    #   within    = rest of the family's diagonal block (member x member)
    #   cross_fam = sum over the other families of F[a,b] + F[b,a]
    main = pd.Series(0.0, index=fam_ids)
    for i, a in enumerate(feats):
        main[fid[a]] += M[i, i]
    diag = pd.Series({f: F.loc[f, f] for f in fam_ids})
    within = diag - main
    cross_fam = pd.Series({f: F.loc[f].sum() + F[f].sum() - 2.0 * F.loc[f, f]
                           for f in fam_ids})
    effects = pd.DataFrame({f"main_effect{u}": main,
                            f"within_family_interaction{u}": within,
                            f"cross_family_interaction{u}": cross_fam})
    effects.round(6).to_csv(out_dir / "interaction_family_effects.csv",
                            index_label="family_id")
    effects.index.name = "family_id"
    _save_pdata(effects, pdata / "interaction_family_effects.csv")

    top_fam = sorted(((F.values[i, j] + F.values[j, i], fam_ids[i],
                       fam_ids[j])
                      for i, j in combinations(range(len(fam_ids)), 2)),
                     reverse=True)
    summary.append(_with_unit("Family main effects, largest first", funit)
                   + ":")
    for f, v in main.sort_values(ascending=False).head(5).items():
        summary.append(f"  {info[f]['text']} = {v*scale:.{dec}f} "
                       f"(+{within[f]*scale:.{dec}f} within-family)")
    summary.append(_with_unit("Strongest cross-family interactions", funit)
                   + ":")
    for v, a, b in top_fam[:5]:
        summary.append(f"  {info[a]['text']} x {info[b]['text']} = "
                       f"{v*scale:.{dec}f}")
    summary.append(f"Totals: main {_num(main.sum()*scale, f'.{dec}f', funit)}"
                   f", within-family "
                   f"{_num(within.sum()*scale, f'.{dec}f', funit)}, "
                   f"cross-family "
                   f"{_num(cross_fam.sum()*scale/2, f'.{dec}f', funit)} "
                   "(cross counted once per pair)")
    return partner, pairs


# =============================================================================
# Dependence and ALE
# =============================================================================
def shap_frame(model, Xs):
    sv = np.asarray(shap.TreeExplainer(model).shap_values(
        Xs, check_additivity=False))
    while sv.ndim > 2:
        sv = sv.mean(axis=-1)
    return pd.DataFrame(sv, columns=Xs.columns, index=Xs.index)


def _direction(rho):
    if np.isnan(rho):
        return "none"
    return "+" if rho > 0 else "-" if rho < 0 else "0"


def ale_1d(predict, Xs, feat, n_bins):
    """1-D accumulated local effects (Apley and Zhu), centred."""
    x = Xs[feat].values
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    K = len(qs) - 1
    if K < 2:
        return None, None
    idx = np.clip(np.searchsorted(qs, x, side="left") - 1, 0, K - 1)

    local = np.zeros(K)
    counts = np.zeros(K, dtype=int)
    for k in range(K):
        m = idx == k
        nk = int(m.sum())
        counts[k] = nk
        if nk == 0:
            continue
        Xl = Xs[m].copy()
        Xl[feat] = qs[k]
        Xu = Xs[m].copy()
        Xu[feat] = qs[k + 1]
        local[k] = float((predict(Xu) - predict(Xl)).mean())

    cum = np.cumsum(local)
    centers = 0.5 * (qs[:-1] + qs[1:])
    w = counts / max(counts.sum(), 1)
    return centers, cum - float((cum * w).sum())


def dependence_ale(xgb, rf, X_train, Xs, sv, key, cfg, out_dir, pdata,
                   summary):
    """Effect directions and ALE curves (figures: plot())."""
    m, rs = cfg["mechanism"], cfg["models"]["random_state"]
    dir_rows = []
    for f in X_train.columns:
        rho, _ = spearmanr(Xs[f], sv[f])
        dir_rows.append({"feature": f,
                         "spearman_x_vs_shap": round(float(rho), 3),
                         "direction": _direction(rho)})
    dirs = pd.DataFrame(dir_rows).set_index("feature")
    dirs.to_csv(out_dir / "effect_directions.csv")
    summary.append("Effect directions (Spearman of x vs SHAP(x), "
                   "key features):")
    for f in key:
        summary.append(f"  {style.short(f)}: {dirs.loc[f, 'direction']} "
                       f"({dirs.loc[f, 'spearman_x_vs_shap']})")

    Xs_ale = X_train.sample(min(m["ale_samples"], len(X_train)),
                            random_state=rs + 1)
    rows = []
    for f in key:
        for tag, model in (("xgboost", xgb), ("randomforest", rf)):
            cx, ay = ale_1d(model.predict, Xs_ale, f, m["ale_bins"])
            if cx is not None:
                rows += [{"feature": f, "model": tag, "x": a, "ale": b}
                         for a, b in zip(cx, ay)]
        print(f"  ALE: {style.short(f)}")
    _save_pdata(pd.DataFrame(rows, columns=["feature", "model", "x", "ale"]),
                pdata / "ale_curves.csv", index=False)


def dependence_directions_by_model(Xs, sv_x, sv_r, out_dir):
    rows = []
    for f in Xs.columns:
        rx, _ = spearmanr(Xs[f], sv_x[f])
        rr, _ = spearmanr(Xs[f], sv_r[f])
        rows.append({"feature": f, "xgb_dir": _direction(rx),
                     "xgb_spearman": round(float(rx), 3),
                     "rf_dir": _direction(rr),
                     "rf_spearman": round(float(rr), 3)})
    pd.DataFrame(rows).set_index("feature").to_csv(
        out_dir / "effect_directions_by_model.csv")


# =============================================================================
# Leave-one-family-out
# =============================================================================
def lofo(X, y, fam, info, cfg, out_dir, pdata, summary):
    unit, scale, funit = _units(cfg)
    u = _suffix(unit)
    dec = 1 if scale == 1000.0 else 4
    m = cfg["mechanism"]
    kf = KFold(n_splits=cfg["models"]["cv_folds"], shuffle=True,
               random_state=cfg["models"]["random_state"])

    def cv_metrics(cols):
        r2s, rmses, maes = [], [], []
        Xc = X[cols]
        for tr, te in kf.split(Xc):
            rf = models.rf_downstream(
                cfg, max_features=min(m["lofo_max_features"], len(cols)))
            rf.fit(Xc.iloc[tr], y.iloc[tr])
            rf.set_params(n_jobs=1)       # fixed summation order: repeatable
            pred = rf.predict(Xc.iloc[te])
            yt = y.iloc[te]
            r2s.append(r2_score(yt, pred))
            rmses.append(float(np.sqrt(mean_squared_error(yt, pred))))
            maes.append(float(mean_absolute_error(yt, pred)))
        return (float(np.mean(r2s)), float(np.mean(rmses)),
                float(np.mean(maes)))

    print("  baseline (all descriptors) ...")
    base_r2, base_rmse, base_mae = cv_metrics(list(X.columns))
    fid = fam["family_id"]
    rows = [{"family": "baseline (all features)", "family_id": "-",
             "members": f"all {X.shape[1]}", "n_removed": 0,
             "cv_r2": base_r2, f"cv_rmse{u}": base_rmse,
             f"cv_mae{u}": base_mae, "dR2": 0.0, f"dRMSE{u}": 0.0,
             f"dMAE{u}": 0.0}]
    for f_id in _sorted_family_ids(fid):
        members = fid.index[fid == f_id].tolist()
        keep = [c for c in X.columns if c not in members]
        r2, rmse, mae = cv_metrics(keep)
        rows.append({"family": info[f_id]["short"], "family_id": f_id,
                     "members": "; ".join(style.short(c) for c in members),
                     "n_removed": len(members), "cv_r2": r2,
                     f"cv_rmse{u}": rmse, f"cv_mae{u}": mae,
                     "dR2": r2 - base_r2, f"dRMSE{u}": rmse - base_rmse,
                     f"dMAE{u}": mae - base_mae})
        print(f"  removed {f_id} ({len(members)}): R2 {r2:.4f} "
              f"(d{r2 - base_r2:+.4f}), RMSE "
              f"{_num(rmse, '.4f', unit)} (d{rmse - base_rmse:+.4f})")
    df = pd.DataFrame(rows)
    df.round(5).to_csv(out_dir / "lofo_ablation.csv", index=False)
    _save_pdata(df, pdata / "lofo_ablation.csv", index=False)

    summary.append(f"LOFO baseline: R2 = {base_r2:.4f}, "
                   f"RMSE = {_num(base_rmse, '.4f', unit)}")
    top = df.iloc[1:].sort_values(f"dRMSE{u}", ascending=False).head(3)
    summary.append("Most expensive families to remove (dRMSE):")
    for _, r in top.iterrows():
        summary.append(f"  {r['family']}: "
                       f"+{_num(r[f'dRMSE{u}'] * scale, f'.{dec}f', funit)} "
                       f"(dR2 {r['dR2']:+.4f})  [{r['members']}]")
    summary.append("(reminder: cross-family correlation means these are "
                   "lower bounds)")


# =============================================================================
# Surrogate trees
# =============================================================================
def _tree_threshold_rows(tree, cols, context, unit):
    u = _suffix(unit)
    t = tree.tree_
    depth = np.zeros(t.node_count, dtype=int)
    stack = [(0, 0)]
    while stack:
        n, d = stack.pop()
        depth[n] = d
        if t.children_left[n] != -1:
            stack.append((t.children_left[n], d + 1))
            stack.append((t.children_right[n], d + 1))
    rows = []
    for n in range(t.node_count):
        L, R = t.children_left[n], t.children_right[n]
        if L == -1:
            continue
        rows.append({
            "context": context,
            "depth": int(depth[n]),
            "split_feature": cols[t.feature[n]],
            "threshold": round(float(t.threshold[n]), 4),
            "n_samples": int(t.n_node_samples[n]),
            f"left_mean{u}": round(float(t.value[L][0][0]), 4),
            f"right_mean{u}": round(float(t.value[R][0][0]), 4),
        })
    return rows


def surrogates(X, y, key, joint_feats, cfg, out_dir, summary, dot=None):
    m = cfg["mechanism"]
    rs = cfg["models"]["random_state"]
    unit = cfg["data"]["target_unit"]
    u = _suffix(unit)
    min_leaf = max(20, int(m["surrogate_min_leaf_frac"] * len(X)))
    thr_rows, drawings = [], []

    for f in key:
        tree = DecisionTreeRegressor(max_depth=m["surrogate_depth_single"],
                                     min_samples_leaf=min_leaf,
                                     random_state=rs)
        Xf = X[[f]]
        tree.fit(Xf, y)
        thr_rows += _tree_threshold_rows(tree, [f],
                                         f"single:{style.short(f)}", unit)
        drawings.append((tree, Xf, f"{outputs.file_stem(f)}_tree.svg"))
        print(f"  surrogate {style.short(f)}: root cutoff = "
              f"{tree.tree_.threshold[0]:.3f}")

    joint = DecisionTreeRegressor(max_depth=m["surrogate_depth_joint"],
                                  min_samples_leaf=min_leaf, random_state=rs)
    Xj = X[joint_feats]
    joint.fit(Xj, y)
    thr_rows += _tree_threshold_rows(joint, joint_feats, "joint", unit)
    drawings.append((joint, Xj, "joint_tree.svg"))

    rules = export_text(joint, feature_names=[style.short(f)
                                              for f in joint_feats])
    with open(out_dir / "joint_tree_rules.txt", "w", encoding="utf-8") as fh:
        fh.write(rules)
    pd.DataFrame(thr_rows).to_csv(out_dir / "surrogate_thresholds.csv",
                                  index=False)

    summary.append(f"Surrogate root cutoffs (single-feature "
                   f"depth-{m['surrogate_depth_single']} trees):")
    for r in thr_rows:
        if r["context"].startswith("single") and r["depth"] == 0:
            summary.append(f"  {r['context'][7:]}: {r['threshold']} "
                           f"(means {r[f'left_mean{u}']} | "
                           f"{_num(r[f'right_mean{u}'], '', unit)})")
    summary.append(f"Joint tree R2 (fit) = {joint.score(Xj, y):.3f} -- "
                   "see joint_tree_rules.txt")

    if dot is not None:
        sur_dir = out_dir / "surrogate_trees"
        sur_dir.mkdir(exist_ok=True)
        print(f"  Graphviz dot: {dot[0]} (version {dot[1]})")
        for tree, Xd, name in drawings:
            _save_viz(tree, Xd, y, str(sur_dir / name), cfg)


def _candidate_dots():
    import glob
    import sys

    exe = "dot.exe" if os.name == "nt" else "dot"
    cands = []
    w = shutil.which("dot")
    if w:
        cands.append(w)
    prefix = sys.prefix
    cands += [
        os.path.join(prefix, "Library", "bin", exe),
        os.path.join(prefix, "Library", "bin", "graphviz", exe),
        os.path.join(prefix, "Scripts", exe),
        os.path.join(prefix, "bin", "dot"),
    ]
    if os.name == "nt":
        cands += glob.glob(os.path.join(prefix, "Library", "bin", "*", exe))
        cands += glob.glob(r"C:\Program Files*\Graphviz*\bin\dot.exe")
    else:
        cands += ["/usr/bin/dot", "/usr/local/bin/dot",
                  "/opt/homebrew/bin/dot"]

    seen, out = set(), []
    for c in cands:
        if c and os.path.isfile(c):
            k = os.path.normcase(os.path.abspath(c))
            if k not in seen:
                seen.add(k)
                out.append(os.path.abspath(c))
    return out


def _register_working_dot():
    """Find a Graphviz dot that the graphviz library can run, and put it on
    PATH. On Windows a dot.bat wrapper (old Anaconda graphviz 2.38) never
    works: Python resolves a bare 'dot' to .exe only."""
    import subprocess
    try:
        import dtreeviz  # noqa: F401
        import graphviz as gv
    except ImportError as exc:
        raise ImportError(
            "Drawing surrogate trees needs dtreeviz (pip install dtreeviz) "
            "and Graphviz. Set mechanism.surrogate_drawings: false to get "
            "the thresholds and rules without drawings.") from exc

    tried = []
    orig_path = os.environ.get("PATH", "")
    for c in _candidate_dots():
        tried.append(c)
        if os.path.splitext(c)[1].lower() in (".bat", ".cmd"):
            continue              # can never satisfy a bare 'dot' call
        try:
            if subprocess.run([c, "-V"], capture_output=True,
                              timeout=15).returncode != 0:
                continue
        except Exception:
            continue
        os.environ["PATH"] = os.path.dirname(c) + os.pathsep + orig_path
        for attempt in (1, 2):
            try:
                try:
                    ver = gv.version()
                except AttributeError:         # very old graphviz package
                    ver = gv.backend.version()
                gv.Source("digraph{a->b}").pipe(format="svg")
                return c, ".".join(map(str, ver))
            except Exception:
                if attempt == 1:
                    try:                       # register plugins, retry once
                        subprocess.run([c, "-c"], capture_output=True,
                                       timeout=15)
                    except Exception:
                        pass
        os.environ["PATH"] = orig_path
    raise RuntimeError(
        "No Graphviz 'dot' that the graphviz library can run was found "
        "(needed to draw surrogate trees). Candidates checked:\n  "
        + ("\n  ".join(tried) if tried else "<none>")
        + "\nA dot.bat wrapper (old Anaconda graphviz 2.38) does not count: "
        "Python resolves a bare 'dot' to .exe only.\nFix, with this "
        "environment active:\n"
        "    conda remove graphviz python-graphviz\n"
        "    conda install -c conda-forge \"graphviz>=8\" python-graphviz\n"
        "then check that 'where.exe dot' shows ...\\Library\\bin\\dot.exe "
        "(.exe, not .bat). Or install Graphviz from graphviz.org and add its "
        "bin folder to PATH. Or set mechanism.surrogate_drawings: false.")


def _save_viz(tree, Xf, yv, path, cfg):
    import dtreeviz
    n = len(Xf)
    sample = cfg["mechanism"]["surrogate_viz_sample"]
    if n > sample:
        idx = np.random.default_rng(cfg["models"]["random_state"]).choice(
            n, sample, replace=False)
        Xv, yvv = Xf.iloc[idx], yv.iloc[idx]
    else:
        Xv, yvv = Xf, yv
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="X does not have valid feature names")
        if hasattr(dtreeviz, "model"):        # dtreeviz >= 2
            vm = dtreeviz.model(tree, X_train=Xv, y_train=yvv,
                                feature_names=list(Xf.columns),
                                target_name=cfg["data"]["target"])
            vm.view(scale=1.2).save(path)
        else:                                  # dtreeviz 1.x
            from dtreeviz.trees import dtreeviz as _dtv
            _dtv(tree, Xv, yvv, target_name=cfg["data"]["target"],
                 feature_names=list(Xf.columns)).save(path)
    stray = path[:-4]                          # dtreeviz leaves its
    if os.path.isfile(stray):                  # intermediate file behind
        os.remove(stray)


# =============================================================================
# Figures
# =============================================================================
def plot_interaction_heatmap(M, feats, unit, out_path):
    order = np.argsort(-M.diagonal())
    fig, ax = plt.subplots(figsize=(10.5, 9))
    Mo = M[np.ix_(order, order)]
    im = ax.imshow(Mo, cmap="viridis")
    labs = [style.short(feats[i]) for i in order]
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_title("Mean |SHAP interaction| (diagonal = main effects)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=unit)
    style.frame_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _fam_fmt(v, cfg):
    """One fixed precision everywhere, so no value carries more apparent
    significance than another."""
    return f"{v:,.{cfg['mechanism']['family_fig_decimals']}f}"


def _fam_save(fig, plots_dir, stem, cfg):
    m = cfg["mechanism"]
    fig.savefig(plots_dir / f"{stem}.png", dpi=m["family_fig_dpi"],
                bbox_inches="tight")
    if m["family_fig_vector"]:
        fig.savefig(plots_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _cell_text_color(cmap, frac):
    r, g, b = cmap(frac)[:3]
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"


def plot_family_main_effects(main, within, info, cfg, plots_dir):
    """The diagonal of the family matrix, as vertical bars. A bar is the pure
    main effect, the sum of M[i,i] over the family's members; with
    main_effect_decompose the within-family interaction is stacked on top."""
    m = cfg["mechanism"]
    _, scale, funit = _units(cfg)
    ids = (list(main.sort_values(ascending=False).index)
           if m["main_effect_sort"] else list(main.index))
    mv = main.reindex(ids).values * scale
    w = (within.reindex(ids).values * scale if m["main_effect_decompose"]
         else np.zeros(len(ids)))
    tot = mv + w
    ref = max(float(tot.max()), 1e-12)
    x = np.arange(len(ids))

    fig, ax = plt.subplots(figsize=(7.2, m["family_fig_height"]))
    ax.bar(x, mv, width=0.68, color="#2c728e", edgecolor="black",
           linewidth=0.6, zorder=3, label="main effect")
    if m["main_effect_decompose"]:
        ax.bar(x, w, bottom=mv, width=0.68, color="#c3cdd4",
               edgecolor="black", linewidth=0.6, zorder=3,
               label="within-family interaction")
        ax.legend(fontsize=8, frameon=False)

    for xx, t in zip(x, tot):
        ax.text(xx, t + 0.015 * ref, _fam_fmt(t, cfg), ha="center",
                va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([info[f]["label"] for f in ids], fontsize=8.5,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(_with_unit("mean |SHAP| attributed within family", funit)
                  if m["main_effect_decompose"] else
                  _with_unit("mean |SHAP| main effect", funit), fontsize=10)
    ax.set_ylim(0, ref * 1.10)
    ax.set_xlim(-0.7, len(ids) - 0.3)
    ax.tick_params(axis="both", length=0, labelsize=8.5)
    if m["main_effect_grid"]:
        ax.grid(alpha=0.3, axis="y", zorder=0)
        ax.set_axisbelow(True)
    style.frame_axes(ax)
    if m["family_fig_titles"]:
        ax.set_title("Family-level main effects", fontsize=11)
    fig.tight_layout()
    _fam_save(fig, plots_dir, "family_main_effects", cfg)


def _plot_family_interaction_panel(F, ids, info, cfg, plots_dir, stem,
                                   boundaries=()):
    """One off-diagonal triangle of the family matrix, in the given order.
    With interaction_pair_sum a cell is F[a,b] + F[b,a], the full strength
    of that family pair, on the same basis as interaction_top_pairs.csv.
    The colour scale is set by the largest cross-family cell. boundaries are
    the block group sizes (cumulative) at which the block outlines go."""
    m = cfg["mechanism"]
    _, scale, funit = _units(cfg)
    n = len(ids)
    V = F.loc[ids, ids].values * scale
    if m["interaction_pair_sum"]:
        V = V + V.T

    lower = m["interaction_triangle"] != "upper"
    A = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if (i > j) if lower else (i < j):
                A[i, j] = V[i, j]
    Am = np.ma.masked_invalid(A)
    vmax = float(Am.max()) if Am.count() else 1.0

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")            # empty triangle reads as absent, not zero

    fig, ax = plt.subplots(figsize=(8.4, m["family_fig_height"]))
    im = ax.imshow(Am, cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal",
                   interpolation="nearest")
    for i in range(n):
        for j in range(n):
            if np.ma.is_masked(Am[i, j]):
                continue
            v = float(Am[i, j])
            ax.text(j, i, _fam_fmt(v, cfg), ha="center", va="center",
                    fontsize=7.5,
                    color=_cell_text_color(cmap, v / vmax if vmax else 0.0))

    labs = [info[f]["label"] for f in ids]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labs, fontsize=8.5, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labs, fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)
    if m["interaction_box"]:
        style.frame_axes(ax)
    else:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    # outline the between-block rectangles of the drawn triangle
    if m["interaction_block_divider"]:
        for count in boundaries:
            if not 0 < count < n:
                continue
            k = count - 0.5
            if lower:
                ax.plot([-0.5, k], [k, k], color="black", lw=1.2, zorder=5)
                ax.plot([k, k], [k, n - 0.5], color="black", lw=1.2,
                        zorder=5)
            else:
                ax.plot([k, n - 0.5], [k, k], color="black", lw=1.2,
                        zorder=5)
                ax.plot([k, k], [-0.5, k], color="black", lw=1.2, zorder=5)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(_with_unit("mean |SHAP interaction|", funit), fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    if m["family_fig_titles"]:
        ax.set_title("Family-level interaction effects (cross-family)",
                     fontsize=11)
    fig.tight_layout()
    _fam_save(fig, plots_dir, stem, cfg)


def plot_family_interactions(F, info, cfg, plots_dir):
    """The cross-family figure, once per entry of interaction_orderings:
    numeric (F1..Fn, as in the dendrogram) or grouped (by block, with the
    between-block rectangles outlined)."""
    fam_ids = list(F.index)
    for how in cfg["mechanism"]["interaction_orderings"]:
        if how == "grouped":
            groups = _family_groups(fam_ids, info,
                                    cfg["labels"]["blocks"])
            if len(groups) < 2:
                print("  note: no blocks to group the families by "
                      "(labels.blocks); skipping the grouped figure")
                continue
            order = [f for g in groups for f in g]
            sizes = np.cumsum([len(g) for g in groups])[:-1]
            _plot_family_interaction_panel(
                F, order, info, cfg, plots_dir,
                "family_interaction_effects_grouped",
                boundaries=[int(s) for s in sizes])
        else:
            _plot_family_interaction_panel(F, fam_ids, info, cfg, plots_dir,
                                           "family_interaction_effects")


def plot_dependence_ale(feat, partner, Xs, sv, curves, unit, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # left: SHAP dependence
    ax = axes[0]
    if partner is not None:
        sc = ax.scatter(Xs[feat], sv[feat], c=Xs[partner], cmap=PARTNER_CMAP,
                        s=8, alpha=0.6)
        fig.colorbar(sc, ax=ax, label=style.short(partner))
    else:
        ax.scatter(Xs[feat], sv[feat], s=8, alpha=0.5)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(style.short(feat))
    ax.set_ylabel(_with_unit("SHAP value", unit))
    ax.set_title("XGBoost SHAP dependence")
    style.frame_axes(ax)

    # right: ALE overlay
    ax = axes[1]
    for tag, ls, label in (("xgboost", "-", "XGBoost"),
                           ("randomforest", "--", "RandomForest")):
        c = curves[(curves["feature"] == feat) & (curves["model"] == tag)]
        if len(c):
            ax.plot(c["x"].to_numpy(), c["ale"].to_numpy(), ls, lw=2,
                    label=label)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(style.short(feat))
    ax.set_ylabel(_with_unit("ALE", unit))
    ax.set_title("Accumulated local effects")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    style.frame_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def dependence_panel(feat, partner, xv, yv, cv, unit, bold, out_path):
    """x = descriptor value, y = its SHAP value (right spine), colour =
    partner value (colorbar on the left). No title."""
    weight = "bold" if bold else "normal"
    fig = plt.figure(figsize=(6.2, 5.0))
    cax = fig.add_axes([0.10, 0.135, 0.032, 0.78])   # left colorbar
    ax = fig.add_axes([0.30, 0.135, 0.60, 0.78])     # scatter

    sc = ax.scatter(xv, yv, c=cv, cmap=PARTNER_CMAP, s=12, alpha=0.65,
                    edgecolors="none")
    ax.axhline(0, color="0.5", lw=1.0, ls="--")
    ax.set_xlabel(style.short(feat), fontsize=15, fontweight=weight)

    # SHAP-value axis on the right
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel(_with_unit("SHAP value", unit), fontsize=15,
                  fontweight=weight, rotation=270, labelpad=20)
    ax.tick_params(axis="both", labelsize=12, colors="black")
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight(weight)
    style.frame_axes(ax, linewidth=1.3)

    cb = fig.colorbar(sc, cax=cax)
    cax.yaxis.set_label_position("left")
    cax.yaxis.set_ticks_position("left")
    cb.set_label(style.short(partner), fontsize=14, fontweight=weight)
    cb.ax.tick_params(labelsize=11)
    for lab in cb.ax.get_yticklabels():
        lab.set_fontweight(weight)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_lofo(df, info, cfg, out_path):
    unit, scale, funit = _units(cfg)
    u = _suffix(unit)
    d = df.iloc[1:].sort_values(f"dRMSE{u}", ascending=False)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(max(8.5, 0.85 * len(d) + 2.5), 6.2))
    ax.bar(x, d[f"dRMSE{u}"] * scale, color="#31688e", edgecolor="black",
           linewidth=0.6, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([info[f]["short"] for f in d["family_id"]])
    style.rotate_xticklabels(ax, fontsize=11)
    ax.set_ylabel(_with_unit("Increase in CV RMSE\nwhen family removed",
                             funit), fontsize=13)
    ax.axhline(0, color="0.6", lw=1.0)
    ax.grid(False)
    ax.set_xlim(-0.7, len(d) - 0.3)
    style.frame_axes(ax, linewidth=1.3)
    ax.tick_params(axis="y", labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Stage entry points
# =============================================================================
def _key_features(cfg, run_dir, feats):
    m = cfg["mechanism"]
    if m["key_features"] == "auto":
        path = run_dir / "consensus" / "consensus_rank_table.csv"
        if not path.exists():
            raise FileNotFoundError(
                "mechanism.key_features is auto, which takes the top "
                "descriptors by consensus rank: run the consensus stage "
                "first, or list key_features.")
        ranks = pd.read_csv(path, index_col=0)
        return [f for f in ranks.index if f in feats][:m["n_key_features"]]
    key = list(m["key_features"])
    _check_names(key, feats, "key_features")
    return key


def run(cfg, X, y, out_dir):
    out_dir = Path(out_dir)
    m = cfg["mechanism"]
    parts = _parts(cfg)
    _units(cfg)                                     # check family_unit early
    run_dir = out_dir.parent
    fam_path = run_dir / "families" / "feature_families.csv"
    if not fam_path.exists():
        raise FileNotFoundError(f"{fam_path} not found; run the families "
                                f"stage first.")
    fam = _read_families(fam_path)
    feats = list(X.columns)
    missing = [c for c in feats if c not in fam.index]
    if missing:
        raise ValueError(f"Descriptors missing from the families stage's "
                         f"output: {missing}. Run the families stage again "
                         f"for this dataset.")
    key = _key_features(cfg, run_dir, feats)
    _check_names(m["partner_report"], feats, "partner_report")
    pairs_cfg = m["pair_plots"]
    if pairs_cfg != "auto":
        for pair in pairs_cfg:
            if len(pair) != 2:
                raise ValueError(f"mechanism.pair_plots: {pair} should be "
                                 f"[descriptor, partner]")
            _check_names(pair, feats, "pair_plots")
    joint = (key[:4] if m["joint_tree_features"] == "auto"
             else list(m["joint_tree_features"]))
    _check_names(joint, feats, "joint_tree_features")
    dot = None
    if "surrogates" in parts and m["surrogate_drawings"]:
        dot = _register_working_dot()          # before any fitting

    outputs.start(out_dir, cfg)
    pdata = out_dir / "plot_data"
    pdata.mkdir()
    _save_pdata(fam, pdata / "feature_families.csv")
    info = _family_info(fam, cfg)
    unit = cfg["data"]["target_unit"]
    rs = cfg["models"]["random_state"]
    summary = []
    print(f"Parts: {', '.join(parts)}")
    print(f"Key descriptors: {', '.join(style.short(k) for k in key)}")

    explain = [p for p in ("interactions", "dependence_ale",
                           "dependence_comparison") if p in parts]
    partner = None
    if explain:
        X_train, y_train = selection_split(X, y, cfg)
        print("Fitting the explanation models on the selection split ...")
        xgb = models.xgb_selector(cfg).fit(X_train, y_train)
        rf = models.rf_selector(cfg).fit(X_train, y_train)
        rf.set_params(n_jobs=1)          # predictions in a fixed order (ALE)

        X_int = X_train.sample(min(m["interaction_samples"], len(X_train)),
                               random_state=rs)
        print(f"SHAP interaction values on {len(X_int)} rows (the slow "
              f"step) ...")
        M = interaction_matrix(xgb, X_int)
        if "interactions" in parts:
            partner, pairs = interactions(M, feats, fam, info, cfg, out_dir,
                                          pdata, summary)
        else:
            partner, _ = strongest_partners(M, feats)
            pairs = pair_table(M, feats, cfg["labels"]["blocks"], unit)

    if "dependence_ale" in parts or "dependence_comparison" in parts:
        Xs = X_train.sample(min(m["shap_samples"], len(X_train)),
                            random_state=rs)
        print(f"SHAP values on {len(Xs)} rows ...")
        sv_x = shap_frame(xgb, Xs)
        _save_pdata(Xs, pdata / "shap_rows.csv")
        _save_pdata(sv_x, pdata / "shap_values_xgboost.csv")
        plan = [{"kind": "key", "feature": f, "partner": partner[f]}
                for f in key]
        if "dependence_comparison" in parts:
            chosen = ([tuple(p) for p in pairs_cfg] if pairs_cfg != "auto"
                      else list(zip(pairs["feature_1"],
                                    pairs["feature_2"]))[:m["n_pair_plots"]])
            plan += [{"kind": "pair", "feature": f, "partner": p}
                     for f, p in chosen]
        _save_pdata(pd.DataFrame(plan, columns=["kind", "feature",
                                                "partner"]),
                    pdata / "dependence_plan.csv", index=False)

        if "dependence_ale" in parts:
            print("SHAP dependence and ALE ...")
            dependence_ale(xgb, rf, X_train, Xs, sv_x, key, cfg, out_dir,
                           pdata, summary)
        if "dependence_comparison" in parts:
            print("RandomForest SHAP values for the dependence comparison "
                  "...")
            sv_r = shap_frame(rf, Xs)
            _save_pdata(sv_r, pdata / "shap_values_randomforest.csv")
            dependence_directions_by_model(Xs, sv_x, sv_r, out_dir)

    if "lofo" in parts:
        print(f"Leave-one-family-out ({cfg['models']['cv_folds']}-fold, "
              f"full data) ...")
        lofo(X, y, fam, info, cfg, out_dir, pdata, summary)

    if "surrogates" in parts:
        print("Surrogate trees ...")
        surrogates(X, y, key, joint, cfg, out_dir, summary, dot)

    header = (f"Parts: {', '.join(parts)}\nKey descriptors: "
              f"{', '.join(style.short(k) for k in key)}")
    (out_dir / "mechanism_summary.txt").write_text(
        header + "\n\n" + "\n".join(summary) + "\n", encoding="utf-8")
    for line in summary:
        print(f"  {line}")


def plot(cfg, out_dir):
    out_dir = Path(out_dir)
    pdata = out_dir / "plot_data"
    if not pdata.exists():
        raise FileNotFoundError(f"{pdata} not found; run the mechanism stage "
                                f"first.")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fam = _read_families(pdata / "feature_families.csv")
    info = _family_info(fam, cfg)
    unit = cfg["data"]["target_unit"]
    u = _suffix(unit)
    bold = cfg["mechanism"]["dependence_bold"]

    def read(name, **kw):
        return _read_pdata(pdata, name, **kw)

    M = read("interaction_matrix.csv", index_col=0)
    effects = read("interaction_family_effects.csv", index_col=0)
    F = read("interaction_family_matrix.csv", index_col=0)
    curves = read("ale_curves.csv")
    plan = read("dependence_plan.csv", keep_default_na=False)
    Xs = read("shap_rows.csv", index_col=0)
    sv_x = read("shap_values_xgboost.csv", index_col=0)
    sv_r = read("shap_values_randomforest.csv", index_col=0)
    lofo_df = read("lofo_ablation.csv", keep_default_na=False)

    with style.default_style():
        if M is not None:
            plot_interaction_heatmap(M.to_numpy(), list(M.columns), unit,
                                     plots_dir / "interaction_heatmap.png")
        if effects is not None:
            plot_family_main_effects(
                effects[f"main_effect{u}"],
                effects[f"within_family_interaction{u}"], info, cfg,
                plots_dir)
        if F is not None:
            plot_family_interactions(F, info, cfg, plots_dir)
        if curves is not None:
            dep_dir = plots_dir / "dependence_ale"
            dep_dir.mkdir(exist_ok=True)
            for _, r in plan[plan["kind"] == "key"].iterrows():
                p = r["partner"] if r["partner"] else None
                stem = outputs.file_stem(r["feature"])
                plot_dependence_ale(r["feature"], p, Xs, sv_x, curves, unit,
                                    dep_dir / f"{stem}.png")
        if lofo_df is not None:
            plot_lofo(lofo_df, info, cfg, plots_dir / "lofo_ablation.png")

    if sv_r is not None:
        comp = plots_dir / "dependence_comparison"
        (comp / "key").mkdir(parents=True, exist_ok=True)
        (comp / "pairs").mkdir(parents=True, exist_ok=True)
        with style.paper_style():
            for _, r in plan.iterrows():
                f, p = r["feature"], r["partner"]
                if r["kind"] == "key":
                    base = comp / "key" / outputs.file_stem(f)
                else:
                    stems = (outputs.file_stem(f), outputs.file_stem(p))
                    base = comp / "pairs" / f"{stems[0]}__by__{stems[1]}"
                for tag, sv in (("xgboost", sv_x), ("randomforest", sv_r)):
                    dependence_panel(f, p, Xs[f].values, sv[f].values,
                                     Xs[p].values, unit, bold,
                                     Path(f"{base}__{tag}.png"))
