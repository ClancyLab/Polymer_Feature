"""Selection stage: BorutaShap, Lasso and XGBoost on one SHAP scale.

Ported from compare_feature_selection.py (selectors, unified SHAP, downstream
RF grid, CV figures), importance_plots.py (importance figures with SEM error
bars) and plot_rmse_from_outputs.py (RMSE panels).

run()   fits the selectors and the RF grid, and writes the tables.
plot()  draws every figure from those tables, so figures can be redrawn
        without refitting:  fsphysics plot CONFIG --stages selection

Tables and figures match the original scripts. Deliberate differences:
  * Error bars are matched to descriptors by name. importance_plots.py
    paired them by position, which put most bars on the wrong descriptor.
  * Error bars come from the same SHAP pass as the points. For Lasso this
    differs slightly from importance_plots.py, which used the whole
    selection split as the SHAP background for its bars.
  * Selected/not-selected markers come from this run's
    selected_feature_sets.csv, not from another output folder.
  * The shadow baseline and the bootstrap/seed error-bar options of
    importance_plots.py, all off in the paper, are not ported.
  * Output names follow the package convention, e.g.
    cv_rmse_vs_max_features.png without the "no_legend_" prefix.
  * The RF grid predicts on one thread, so its results repeat exactly. The
    original's multithreaded predictions vary in the last digits (about
    1e-15) between runs on a multi-core machine.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from . import models, outputs, style
from ._vendor.borutashap import BorutaShap
from .data import fingerprint, selection_split

USES_MODELS = True         # runs with the tuned models when tuning applies


# =============================================================================
# SHAP helpers: mean |SHAP| (target units) and its standard error
# =============================================================================
def _shap_rows(X, cfg):
    """Rows used for SHAP values (subsampled if the frame is large)."""
    n = cfg["models"]["shap_max_samples"]
    if len(X) > n:
        return X.sample(n, random_state=cfg["models"]["random_state"])
    return X


def _mean_sem(abs_sv, columns):
    """Mean |SHAP| per descriptor and the standard error of that mean,
    both labelled by descriptor name so they can never be misaligned."""
    n = abs_sv.shape[0]
    mean = pd.Series(abs_sv.mean(axis=0), index=columns)
    sem = pd.Series(abs_sv.std(axis=0, ddof=1) / np.sqrt(n), index=columns)
    return mean, sem


def tree_shap(model, X, cfg):
    """Mean |SHAP| and SEM for a tree model (RandomForest / XGBoost)."""
    Xs = _shap_rows(X, cfg)
    sv = shap.TreeExplainer(model).shap_values(Xs, check_additivity=False)
    sv = np.abs(np.asarray(sv))
    while sv.ndim > 2:                    # collapse a trailing output axis
        sv = sv.mean(axis=-1)
    return _mean_sem(sv, X.columns)


def linear_shap(model, X_scaled, columns, cfg):
    """Mean |SHAP| and SEM for Lasso (exact LinearExplainer)."""
    Xs = X_scaled
    n = cfg["models"]["shap_max_samples"]
    if len(Xs) > n:
        rs = np.random.RandomState(cfg["models"]["random_state"])
        Xs = Xs[rs.choice(len(Xs), n, replace=False)]
    sv = np.asarray(shap.LinearExplainer(model, Xs).shap_values(Xs))
    return _mean_sem(np.abs(sv), columns)


# =============================================================================
# Selectors. Each returns (selected, native, shap_mean, shap_sem); the last
# three are Series indexed by every descriptor.
# =============================================================================
def select_borutashap(X_train, y_train, cfg, out_dir):
    selector = BorutaShap(model=models.rf_selector(cfg),
                          importance_measure="shap", classification=False)
    # fit random_state left at BorutaShap's default (0), as in the paper run
    selector.fit(X_train, y_train, n_trials=cfg["selection"]["boruta_trials"],
                 sample=False, train_or_test="train")
    selector.results_to_csv(
        filename=os.path.join(out_dir, "borutashap_feature_importance"))

    # `accepted` comes back in set order, which changes between sessions;
    # the RF grid depends on column order, so use the dataset order.
    accepted = set(selector.accepted)
    selected = [c for c in X_train.columns if c in accepted]

    # NATIVE: trial-averaged z-scored |SHAP|, skipping the zero-init row 0
    # and the shadow-statistic columns.
    try:
        shadow = {"Max_Shadow", "Min_Shadow", "Mean_Shadow", "Median_Shadow"}
        real = [c for c in selector.history_x.columns if c not in shadow]
        native = selector.history_x[real].iloc[1:].mean(axis=0)
        native = native.reindex(X_train.columns)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Could not read BorutaShap importance history ({exc}); "
                      "native importance set to NaN. Selection is unaffected.")
        native = pd.Series(np.nan, index=X_train.columns)

    # UNIFIED SHAP: a fresh copy of the same forest, explained directly.
    rf = models.rf_selector(cfg).fit(X_train, y_train)
    mean, sem = tree_shap(rf, X_train, cfg)
    return selected, native, mean, sem


def select_lasso(X_train, y_train, cfg):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lasso = models.lasso_cv(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # convergence chatter
        lasso.fit(X_scaled, y_train)
    native = pd.Series(np.abs(lasso.coef_), index=X_train.columns)
    selected = native[native > 1e-8].index.tolist()
    mean, sem = linear_shap(lasso, X_scaled, X_train.columns, cfg)
    return selected, native, mean, sem


def _gain_threshold(native, spec):
    """Parse a SelectFromModel-style threshold spec against gain values."""
    if isinstance(spec, (int, float)):
        return float(spec)
    s = str(spec).replace(" ", "")
    scale, _, base = s.rpartition("*")
    scale = float(scale) if scale else 1.0
    if base == "mean":
        return scale * float(np.mean(native.values))
    if base == "median":
        return scale * float(np.median(native.values))
    raise ValueError(f"Unrecognised xgb_selection: {spec!r}")


def select_xgboost(X_train, y_train, cfg):
    """Top-k by mean |SHAP| (frozen protocol) or a gain threshold."""
    model = models.xgb_selector(cfg)
    model.fit(X_train, y_train)
    native = pd.Series(model.feature_importances_, index=X_train.columns)
    mean, sem = tree_shap(model, X_train, cfg)
    mode = cfg["selection"]["xgb_selection"]
    if mode == "shap_topk":
        k = min(cfg["selection"]["xgb_top_k"], X_train.shape[1])
        selected = mean.sort_values(ascending=False).head(k).index.tolist()
    else:
        thr = _gain_threshold(native, mode)
        selected = native[native >= thr].index.tolist()
    return selected, native, mean, sem


METHODS = ("borutashap", "lasso", "xgboost")


def _select(method, X_train, y_train, cfg, out_dir):
    if method == "borutashap":
        return select_borutashap(X_train, y_train, cfg, out_dir)
    if method == "lasso":
        return select_lasso(X_train, y_train, cfg)
    if method == "xgboost":
        return select_xgboost(X_train, y_train, cfg)
    raise KeyError(f"Unknown selection method '{method}'")


# =============================================================================
# Downstream RandomForest: K-fold CV over a max_features sweep
# =============================================================================
def _residual_plots(preds, residuals, name, folder):
    with style.default_style():
        plt.figure(figsize=(6, 5))
        plt.scatter(preds, residuals, alpha=0.6)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{name}_residuals_vs_pred.png"),
                    dpi=300)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.hist(residuals, bins=30)
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{name}_residual_hist.png"), dpi=300)
        plt.close()


def run_rf_grid(X, y, selected, method, cfg, out_dir):
    if len(selected) == 0:
        print(f"  [{method}] no features selected -- skipping RF grid.")
        return pd.DataFrame()

    m, s = cfg["models"], cfg["selection"]
    p = m["rf_downstream"]
    kf = KFold(n_splits=m["cv_folds"], shuffle=True,
               random_state=m["random_state"])

    # max_features can't exceed the number of selected descriptors; cap and
    # de-duplicate so identical models aren't run twice.
    n_sel = len(selected)
    full = s["max_features_full"] or X.shape[1]
    max_feature_values = sorted({min(f, n_sel) for f in range(1, full + 1)})

    resid_dir = os.path.join(out_dir, "residual_plots", method)
    if s["residual_plots"]:
        os.makedirs(resid_dir, exist_ok=True)

    results = []
    for f in max_feature_values:
        fold_scores, fold_mse, fold_rmse, fold_mae = [], [], [], []
        all_residuals, all_preds = [], []
        for tr, te in kf.split(X):
            model = models.rf_downstream(cfg, max_features=f)
            model.fit(X.iloc[tr][selected], y.iloc[tr])
            # Predict on one thread so the trees are summed in a fixed order.
            # With n_jobs=-1 they are summed in whatever order the threads
            # finish, which changes the last digits from run to run on a
            # multi-core machine.
            model.set_params(n_jobs=1)
            preds = model.predict(X.iloc[te][selected])
            y_te = y.iloc[te]
            fold_scores.append(r2_score(y_te, preds))
            mse = float(mean_squared_error(y_te, preds))
            fold_mse.append(mse)
            fold_rmse.append(float(np.sqrt(mse)))
            fold_mae.append(float(mean_absolute_error(y_te, preds)))
            all_preds.extend(preds)
            all_residuals.extend(y_te - preds)

        name = (f"{method}_RF_n{p.get('n_estimators')}_d{p.get('max_depth')}"
                f"_leaf{p.get('min_samples_leaf')}_feat{f}"
                f"_split{p.get('min_samples_split')}").replace(".", "p")
        if s["residual_plots"] and f <= s["max_features_main"]:
            _residual_plots(all_preds, all_residuals, name, resid_dir)

        results.append({
            "method": method,
            "n_estimators": p.get("n_estimators"),
            "max_depth": p.get("max_depth"),
            "min_samples_leaf": p.get("min_samples_leaf"),
            "max_features": f,
            "min_samples_split": p.get("min_samples_split"),
            "n_selected_features": n_sel,
            "cv_mean": np.mean(fold_scores),
            "cv_std": np.std(fold_scores),
            "cv_mse_mean": np.mean(fold_mse),
            "cv_mse_std": np.std(fold_mse),
            "cv_rmse_mean": np.mean(fold_rmse),
            "cv_rmse_std": np.std(fold_rmse),
            "cv_mae_mean": np.mean(fold_mae),
            "cv_mae_std": np.std(fold_mae),
            "fold_scores": fold_scores,
        })
    return pd.DataFrame(results)


# =============================================================================
# Comparison tables (unchanged from compare_feature_selection.py except the
# unit names in the summary)
# =============================================================================
def build_feature_set_matrix(selections, all_features):
    matrix = pd.DataFrame(index=sorted(all_features))
    for method, feats in selections.items():
        matrix[method] = matrix.index.isin(feats)
    matrix["n_methods_selecting"] = matrix.sum(axis=1)
    matrix = matrix.sort_values("n_methods_selecting", ascending=False)
    return matrix


def build_importance_comparison(importances, selections):
    comp = pd.DataFrame(importances)
    comp = comp.add_suffix("_importance")
    for method, scores in importances.items():
        s = scores.astype(float)
        rng = s.max() - s.min()
        comp[f"{method}_norm"] = (s - s.min()) / rng if rng and rng > 0 else 0.0
        comp[f"{method}_rank"] = s.rank(ascending=False, method="min")
        comp[f"{method}_selected"] = pd.Series(
            s.index.isin(selections[method]), index=s.index)
    norm_cols = [c for c in comp.columns if c.endswith("_norm")]
    comp["mean_norm_importance"] = comp[norm_cols].mean(axis=1)
    comp = comp.sort_values("mean_norm_importance", ascending=False)
    return comp


def build_summary(grid_results, selections, cfg):
    """One row per method: best CV R2 and the config that reached it."""
    unit = cfg["data"]["target_unit"]
    conv = cfg["data"]["unit_conversions"]
    u = f"_{unit}" if unit else ""
    keys = (["best_cv_r2_mean", "best_cv_r2_std",
             f"best_mse{u}2" if unit else "best_mse",
             f"best_rmse{u}", f"best_mae{u}"]
            + [f"best_{k}_{c}" for c in conv for k in ("rmse", "mae")]
            + ["best_max_features"])
    rows = []
    for method in selections:
        df = grid_results.get(method, pd.DataFrame())
        row = {"method": method,
               "n_selected_features": len(selections[method]),
               "selected_features": ", ".join(sorted(selections[method]))}
        if not df.empty:
            best = df.loc[df["cv_mean"].idxmax()]
            rmse, mae = float(best["cv_rmse_mean"]), float(best["cv_mae_mean"])
            vals = [round(float(best["cv_mean"]), 4),
                    round(float(best["cv_std"]), 4),
                    round(float(best["cv_mse_mean"]), 6),
                    round(rmse, 4), round(mae, 4)]
            for factor in conv.values():
                vals += [round(rmse * factor, 3), round(mae * factor, 3)]
            vals.append(int(best["max_features"]))
        else:
            vals = [np.nan] * len(keys)
        row.update(dict(zip(keys, vals)))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Figures from compare_feature_selection.py (matplotlib default style)
# =============================================================================
def plot_importance_heatmap(comp, out_path, title):
    norm_cols = [c for c in comp.columns if c.endswith("_norm")]
    methods = [c.replace("_norm", "") for c in norm_cols]
    mat = comp[norm_cols].copy()
    mat.columns = methods

    fig, ax = plt.subplots(
        figsize=(1.6 * len(methods) + 3, max(4, 0.32 * len(mat))))
    im = ax.imshow(mat.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.5 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="normalised importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_selection_membership(matrix, out_path):
    method_cols = [c for c in matrix.columns if c != "n_methods_selecting"]
    mat = matrix[method_cols].astype(int)

    fig, ax = plt.subplots(
        figsize=(1.6 * len(method_cols) + 3, max(4, 0.32 * len(mat))))
    ax.imshow(mat.values, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(method_cols)))
    ax.set_xticklabels(method_cols)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_title("Selected (filled) vs not selected, by method")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cv_vs_max_features(grid_results, out_path, mean_col, std_col, ylabel):
    """One CV metric (mean +/- std over folds) against max_features."""
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D"]
    x_values = set()
    for i, (method, df) in enumerate(grid_results.items()):
        if df.empty:
            continue
        d = df.sort_values("max_features")
        ax.errorbar(d["max_features"], d[mean_col], yerr=d[std_col],
                    marker=markers[i % len(markers)],
                    linestyle=styles[i % len(styles)],
                    capsize=3, alpha=0.8,
                    label=style.METHOD_LABELS.get(method, method))
        x_values.update(int(v) for v in d["max_features"])
    if x_values:
        ax.set_xticks(sorted(x_values))
    ax.set_xlabel("Max features")
    ax.set_ylabel(ylabel)
    ax.grid(False)
    style.frame_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _cv_metrics(cfg):
    """key -> (mean column, std column, y label); RMSE added from
    plot_rmse_from_outputs.py."""
    k, unit = cfg["models"]["cv_folds"], cfg["data"]["target_unit"]
    u = f" ({unit})" if unit else ""
    u2 = f" ({unit}\u00b2)" if unit else ""
    return {
        "r2": ("cv_mean", "cv_std", f"{k}-fold CV R\u00b2"),
        "rmse": ("cv_rmse_mean", "cv_rmse_std", f"{k}-fold CV RMSE{u}"),
        "mae": ("cv_mae_mean", "cv_mae_std", f"{k}-fold CV MAE{u}"),
        "mse": ("cv_mse_mean", "cv_mse_std", f"{k}-fold CV MSE{u2}"),
    }


# =============================================================================
# Importance figures from importance_plots.py (paper style)
# =============================================================================
def apply_scale(mean_v, se_v, scale, unit):
    """Values, errors and y label on the requested scale."""
    if scale == "raw":
        return mean_v, se_v, f"Mean |SHAP| ({unit})" if unit else "Mean |SHAP|"
    if scale == "fraction":
        tot = mean_v.sum()
        return (100.0 * mean_v / tot, 100.0 * se_v / tot,
                "Share of total importance (%)")
    if scale == "minmax":
        lo, hi = mean_v.min(), mean_v.max()
        if hi - lo <= 0:
            return mean_v * 0, se_v * 0, "Normalised importance"
        return (mean_v - lo) / (hi - lo), se_v / (hi - lo), \
            "Normalised importance"
    raise ValueError(f"Unknown importance_scale: {scale!r}")


def plot_single(method, feats, vals, errs, ylab, out_path, selected=None,
                top_n=8, errorbar_k=1.0):
    order = np.argsort(-vals)
    if top_n:
        order = order[:top_n]
    f = [style.short(feats[i]) for i in order]
    v, e = vals[order], errs[order]
    x = np.arange(len(f))
    col = style.METHOD_COLORS[method]

    fig, ax = plt.subplots(figsize=(max(8.5, 0.62 * len(f) + 3.2), 6.4))
    has_err = np.any(e > 0)
    if selected is None:
        ax.errorbar(x, v, yerr=(errorbar_k * e) if has_err else None,
                    fmt="o", ms=8, color=col, markeredgecolor="black",
                    markeredgewidth=0.8, ecolor="black", elinewidth=1.4,
                    capsize=4, capthick=1.4, linestyle="none", zorder=3)
    else:
        sel = selected[order]
        for mask, face, lab in ((sel, col, "selected"),
                                (~sel, "white", "not selected")):
            if not mask.any():
                continue
            ax.errorbar(x[mask], v[mask],
                        yerr=(errorbar_k * e[mask]) if has_err else None,
                        fmt="o", ms=8, markerfacecolor=face,
                        markeredgecolor=col, markeredgewidth=1.6,
                        ecolor="black", elinewidth=1.4, capsize=4,
                        capthick=1.4, linestyle="none", label=lab, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(f)
    style.rotate_xticklabels(ax, rotation=45, fontsize=11)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_xlabel("Feature", fontsize=14)
    ax.grid(False)
    ax.axhline(0, color="0.75", lw=1.0, zorder=1)
    ax.set_xlim(-0.7, len(f) - 0.3)
    style.frame_axes(ax, linewidth=1.3)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(feats, per_method, ylab, out_path, order_by, top_n=8,
                    mode="union", errorbar_k=1.0):
    order = np.argsort(-per_method[order_by][0])
    if top_n:
        if mode == "union":
            # every descriptor that is top-n for ANY method, laid out in the
            # ordering method's ranking
            keep = set()
            for m in per_method:
                keep.update(np.argsort(-per_method[m][0])[:top_n].tolist())
            order = np.array([i for i in order if i in keep])
        else:
            order = order[:top_n]
    f = [style.short(feats[i]) for i in order]
    x = np.arange(len(f))
    methods = list(per_method.keys())
    off = np.linspace(-0.24, 0.24, len(methods))

    fig, ax = plt.subplots(figsize=(max(9.5, 0.72 * len(f) + 3.5), 6.8))
    for m, dx in zip(methods, off):
        v, e = per_method[m]
        ax.errorbar(x + dx, v[order], yerr=errorbar_k * e[order], fmt="o",
                    ms=7, color=style.METHOD_COLORS[m],
                    markeredgecolor="black", markeredgewidth=0.7,
                    ecolor="black", elinewidth=1.1, capsize=3, capthick=1.1,
                    linestyle="none", label=style.METHOD_LABELS[m], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(f)
    style.rotate_xticklabels(ax, rotation=45, fontsize=11)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_xlabel("Feature", fontsize=14)
    ax.grid(False)
    ax.axhline(0, color="0.75", lw=1.0, zorder=1)
    ax.set_xlim(-0.7, len(f) - 0.3)
    style.frame_axes(ax, linewidth=1.3)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=13, title="Selection method", title_fontsize=13,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _importance_figures(cfg, out_dir, plots_dir):
    """Importance figures and importance_values.csv (importance_plots.py).

    Points are the mean |SHAP| values of this run; error bars are their
    standard errors from the same SHAP pass, matched to the points by
    descriptor NAME. (importance_plots.py paired them by position, which put
    most bars on the wrong descriptor.) The selected/not-selected markers come
    from this run's selected_feature_sets.csv.
    """
    s, unit = cfg["selection"], cfg["data"]["target_unit"]
    top_n = s["top_n"]
    shap_comp = pd.read_csv(os.path.join(
        out_dir, "feature_importance_shap_comparison.csv"), index_col=0)
    sem = pd.read_csv(os.path.join(out_dir, "feature_importance_shap_sem.csv"),
                      index_col=0)
    sets = pd.read_csv(os.path.join(out_dir, "selected_feature_sets.csv"),
                       index_col=0)
    feats = list(shap_comp.index)
    methods = [m for m in s["methods"] if f"{m}_importance" in shap_comp]
    sizes = [(top_n, "")] + ([(None, "_all")] if top_n else [])
    u = f"_{unit}" if unit else ""

    rows, per_method = [], {}
    for method in methods:
        mean_v = shap_comp[f"{method}_importance"].to_numpy(float)
        if s["error_bars"] == "sem":
            err_v = sem[method].reindex(feats).to_numpy(float)     # by name
        else:
            err_v = np.zeros(len(feats))
        vals, errs, ylab = apply_scale(mean_v, err_v, s["importance_scale"],
                                       unit)
        per_method[method] = (vals, errs)

        selected = None
        if method in s["mark_selection"] and method in sets:
            selected = (sets[method].reindex(feats).fillna(False)
                        .astype(bool).to_numpy())
        for n, suffix in sizes:
            plot_single(method, feats, vals, errs, ylab,
                        os.path.join(plots_dir,
                                     f"importance_{method}{suffix}.png"),
                        selected=selected, top_n=n,
                        errorbar_k=s["errorbar_k"])

        rank = pd.Series(mean_v, index=feats).rank(ascending=False,
                                                   method="min")
        tot, lo, hi = mean_v.sum(), mean_v.min(), mean_v.max()
        top = rank.sort_values().index[0]
        print(f"[{method}] top descriptor: {style.short(top)} "
              f"({100 * mean_v[feats.index(top)] / tot:.1f}% of total)")
        for i, f_ in enumerate(feats):
            rec = {"feature": f_, "method": method,
                   f"mean_abs_shap{u}": round(float(mean_v[i]), 6),
                   "fraction_pct": round(100.0 * mean_v[i] / tot, 3),
                   "minmax": round(float((mean_v[i] - lo) / (hi - lo))
                                   if hi > lo else 0.0, 4),
                   "rank": int(rank[f_])}
            if top_n:
                rec["in_top_n_figure"] = bool(rank[f_] <= top_n)
            if np.any(err_v > 0):
                rec[f"error{u}"] = round(float(err_v[i]), 6)
                rec["error_source"] = s["error_bars"]
            rows.append(rec)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "importance_values.csv"),
                              index=False)

    if not per_method:
        return
    _, _, ylab = apply_scale(np.array([1.0, 2.0]), np.array([0.1, 0.1]),
                             s["importance_scale"], unit)
    order_by = s["comparison_order_by"]
    if order_by not in per_method:
        order_by = methods[0]
    for n, suffix in sizes:
        plot_comparison(feats, per_method, ylab,
                        os.path.join(plots_dir,
                                     f"importance_comparison{suffix}.png"),
                        order_by=order_by, top_n=n,
                        mode=s["comparison_top_n"],
                        errorbar_k=s["errorbar_k"])


# =============================================================================
# Stage entry points
# =============================================================================
def run(cfg, X, y, out_dir):
    out_dir = str(out_dir)
    s = cfg["selection"]
    unknown = [m for m in s["methods"] if m not in METHODS]
    if unknown:
        raise KeyError(f"selection.methods: unknown {unknown}; choose from "
                       f"{list(METHODS)}")
    X_train, y_train = selection_split(X, y, cfg)
    outputs.start(out_dir, cfg)
    print(f"Selection split: {len(X_train)} rows "
          f"({100 * cfg['models']['selection_fraction']:.0f}% of the data)")
    with open(os.path.join(out_dir, "split_info.json"), "w") as fh:
        json.dump({"n_rows": int(len(X_train)),
                   "data_sha256": fingerprint(X_train, y_train),
                   "selection_fraction": cfg["models"]["selection_fraction"],
                   "random_state": cfg["models"]["random_state"]},
                  fh, indent=1)

    selections, native, shap_mean, shap_sem = {}, {}, {}, {}
    for i, method in enumerate(s["methods"], start=1):
        print(f"[{i}/{len(s['methods'])}] "
              f"{style.METHOD_LABELS.get(method, method)} ...")
        (selections[method], native[method], shap_mean[method],
         shap_sem[method]) = _select(method, X_train, y_train, cfg, out_dir)
        print(f"      selected {len(selections[method])} descriptors")

    # One RF grid over the full max_features sweep; the main outputs are its
    # rows with max_features <= max_features_main, the *_all outputs all rows.
    grids, grids_all = {}, {}
    for method, selected in selections.items():
        print(f"RF grid for {method} ...")
        g_all = run_rf_grid(X, y, selected, method, cfg, out_dir)
        g = g_all
        if not g_all.empty:
            g = g_all[g_all["max_features"] <= s["max_features_main"]]
            g = g.reset_index(drop=True)
            g.to_csv(os.path.join(out_dir, f"{method}_grid_results.csv"),
                     index=False)
            g_all.to_csv(os.path.join(out_dir,
                                      f"{method}_grid_results_all.csv"),
                         index=False)
        grids[method], grids_all[method] = g, g_all

    p = lambda name: os.path.join(out_dir, name)        # noqa: E731
    build_feature_set_matrix(selections, list(X.columns)).to_csv(
        p("selected_feature_sets.csv"))
    build_importance_comparison(native, selections).to_csv(
        p("feature_importance_comparison.csv"))
    shap_comp = build_importance_comparison(shap_mean, selections)
    shap_comp.to_csv(p("feature_importance_shap_comparison.csv"))
    pd.DataFrame(shap_sem).to_csv(p("feature_importance_shap_sem.csv"))
    summary = build_summary(grids, selections, cfg)
    summary.to_csv(p("method_summary.csv"), index=False)
    summary_all = build_summary(grids_all, selections, cfg)
    summary_all.to_csv(p("method_summary_all.csv"), index=False)

    print("\nBest CV R\u00b2 (max_features <= "
          f"{s['max_features_main']} | full sweep):")
    for (_, a), (_, b) in zip(summary.iterrows(), summary_all.iterrows()):
        print(f"  {a['method']:>10s}: {a['best_cv_r2_mean']:.4f} at "
              f"{a['best_max_features']} | {b['best_cv_r2_mean']:.4f} at "
              f"{b['best_max_features']}")


def plot(cfg, out_dir):
    out_dir = str(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    s = cfg["selection"]

    def read(name):
        path = os.path.join(out_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    grids = {m: read(f"{m}_grid_results.csv") for m in s["methods"]}
    grids_all = {m: read(f"{m}_grid_results_all.csv") for m in s["methods"]}

    with style.default_style():
        plot_importance_heatmap(
            pd.read_csv(os.path.join(out_dir,
                        "feature_importance_comparison.csv"), index_col=0),
            os.path.join(plots_dir, "feature_importance_native_heatmap.png"),
            title="Native importance by method (per-method normalised)")
        plot_importance_heatmap(
            pd.read_csv(os.path.join(out_dir,
                        "feature_importance_shap_comparison.csv"),
                        index_col=0),
            os.path.join(plots_dir, "feature_importance_shap_heatmap.png"),
            title="Unified SHAP importance by method (per-method normalised)")
        plot_selection_membership(
            pd.read_csv(os.path.join(out_dir, "selected_feature_sets.csv"),
                        index_col=0),
            os.path.join(plots_dir, "selection_membership.png"))
        for key, (mean_col, std_col, ylabel) in _cv_metrics(cfg).items():
            for results, suffix in ((grids, ""), (grids_all, "_all")):
                plot_cv_vs_max_features(
                    results,
                    os.path.join(plots_dir,
                                 f"cv_{key}_vs_max_features{suffix}.png"),
                    mean_col, std_col, ylabel)

    with style.paper_style():
        _importance_figures(cfg, out_dir, plots_dir)
