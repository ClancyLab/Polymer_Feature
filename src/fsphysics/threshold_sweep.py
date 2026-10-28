"""Threshold-sweep stage: how the XGBoost cut changes downstream accuracy.

Ported from xgboost_diagnostics.py. It separates XGBoost the model from
XGBoost the selector:

  1. XGBoost model: K-fold CV R2 on all descriptors with the selection-stage
     settings (models.xgb).
  2. Tuning: a randomized search over XGBoost settings, then the CV R2 of the
     tuned model on all descriptors.
  3. Feature-count sweep: rank the descriptors with the tuned model on the
     selection split, by gain and by mean |SHAP|, then score the downstream
     RF on the top k for each k. The paper chose k = 8 from the plateau of
     this curve.
  4. Threshold comparison: SelectFromModel gain thresholds and fixed top-k
     cuts, with the number kept and the downstream RF R2 for each.

Each downstream score is the best K-fold CV R2 over a few max_features
values (1, sqrt(p), 5 and 10, capped at p), as in the original script.

Differences from xgboost_diagnostics.py, all deliberate:
  * The red line marks the cut a gain threshold (mark_cut, default "mean")
    gives with the selection-stage XGBoost, computed from the data. The
    script drew it at 3, the paper's value. cut_label sets its text; the
    paper config keeps "original cut".
  * No in-plot title or grid (figures are described in captions); closed
    box.
  * Model scores, tuned settings and the cut are saved to files, not only
    printed.
  * The downstream RF predicts on one thread, so results repeat exactly,
    and a descriptor list scored twice is computed once.
  * The sweep always includes k = all descriptors; search_iter: 0 skips
    the tuning; groups for GroupKFold come from data.group_column.

run()   fits and writes the tables; plot() draws the figure from them.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint, uniform
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import (GroupKFold, KFold, RandomizedSearchCV,
                                     cross_val_score)
from xgboost import XGBRegressor

from . import models, outputs, style
from .data import load_column, selection_split
from .selection import _gain_threshold, tree_shap

USES_MODELS = True         # runs with the tuned models when tuning applies

_DISTRIBUTIONS = {"randint": randint, "uniform": uniform,
                  "loguniform": loguniform}


# =============================================================================
# Helpers
# =============================================================================
def _cv(cfg, n_splits, groups):
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    return KFold(n_splits=n_splits, shuffle=True,
                 random_state=cfg["models"]["random_state"])


def _search_space(spec):
    """Config search space -> scipy distributions (or lists of values)."""
    space = {}
    for name, value in spec.items():
        if (isinstance(value, (list, tuple)) and value
                and value[0] in _DISTRIBUTIONS):
            space[name] = _DISTRIBUTIONS[value[0]](*value[1:])
        elif isinstance(value, (list, tuple)) and value:
            space[name] = list(value)                    # explicit choices
        else:
            raise ValueError(f"threshold_sweep.search_space.{name}: use "
                             f"[randint|uniform|loguniform, a, b] or a list "
                             f"of values")
    return space


def _plain(value):
    """numpy scalar -> plain Python number, for JSON."""
    return value.item() if hasattr(value, "item") else value


def xgb_cv_r2(params, X, y, groups, cfg):
    """K-fold CV R2 of an XGBRegressor on all descriptors."""
    model = XGBRegressor(**{**params, "n_jobs": -1})
    scores = cross_val_score(model, X, y,
                             cv=_cv(cfg, cfg["models"]["cv_folds"], groups),
                             groups=groups, scoring="r2", n_jobs=-1)
    return scores.mean(), scores.std()


def tune_xgb(X, y, groups, cfg):
    """Randomized search over XGBoost settings; (best settings, best score)."""
    t = cfg["threshold_sweep"]
    base = XGBRegressor(random_state=cfg["models"]["random_state"], n_jobs=1,
                        importance_type="gain")
    search = RandomizedSearchCV(
        base, param_distributions=_search_space(t["search_space"]),
        n_iter=t["search_iter"], scoring="r2",
        cv=_cv(cfg, t["search_folds"], groups),
        random_state=cfg["models"]["random_state"], n_jobs=-1, verbose=0)
    fit_kw = {"groups": groups} if groups is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y, **fit_kw)
    return search.best_params_, search.best_score_


def _rf_scorer(X, y, groups, cfg):
    """best_rf_cv(feats) -> (best CV R2, its max_features) for the downstream
    RF on a descriptor list. Results are cached by the ordered list, since
    column order changes a forest's scores."""
    cache = {}
    cv = _cv(cfg, cfg["models"]["cv_folds"], groups)

    def best_rf_cv(feats):
        feats = list(feats)
        key = tuple(feats)
        if key in cache:
            return cache[key]
        if len(feats) == 0:
            return np.nan, None
        Xf = X[feats]
        p = len(feats)
        candidates = sorted({1, max(1, round(p ** 0.5)), min(p, 5),
                             min(p, 10)})
        best_r2, best_mf = -np.inf, None
        for mf in candidates:
            fold = []
            for tr, te in cv.split(Xf, y, groups):
                m = models.rf_downstream(cfg, max_features=mf)
                m.fit(Xf.iloc[tr], y.iloc[tr])
                m.set_params(n_jobs=1)    # fixed summation order: repeatable
                fold.append(r2_score(y.iloc[te], m.predict(Xf.iloc[te])))
            mean = float(np.mean(fold))
            if mean > best_r2:
                best_r2, best_mf = mean, mf
        cache[key] = (best_r2, best_mf)
        return cache[key]

    return best_rf_cv


def _cut_label(cfg, cut_k):
    t = cfg["threshold_sweep"]
    if t["cut_label"]:
        text = t["cut_label"]
    elif isinstance(t["mark_cut"], (int, np.integer)) \
            and not isinstance(t["mark_cut"], bool):
        text = "chosen cut"
    else:
        text = f"gain \u2265 {t['mark_cut']}"
    return f"{text} ({cut_k})"


# =============================================================================
# Figure
# =============================================================================
def plot_feature_count_sweep(sweep, summary, cfg, out_path):
    """Downstream RF R2 against the number of top XGBoost descriptors kept."""
    n_feat = int(summary["n_features"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(sweep["k"], sweep["r2_gain_ranked"], "o-",
            label="RF on top-k (gain-ranked)")
    ax.plot(sweep["k"], sweep["r2_shap_ranked"], "s--",
            label="RF on top-k (SHAP-ranked)")
    ax.axhline(summary["rf_all_r2"], color="grey", ls=":",
               label=f"RF on all {n_feat} features")
    ax.axhline(summary["xgb_r2_mean"], color="green", ls="-.", alpha=0.7,
               label="XGBoost model (all features)")
    if pd.notna(summary["cut_k"]):
        cut_k = int(summary["cut_k"])
        ax.axvline(cut_k, color="red", alpha=0.4)
        ax.text(cut_k + 0.1, ax.get_ylim()[0], _cut_label(cfg, cut_k),
                color="red", fontsize=8, va="bottom")
    ax.set_xlabel("number of top XGBoost features kept (k)")
    ax.set_ylabel(f"{cfg['models']['cv_folds']}-fold CV R\u00b2 "
                  f"(best over max_features)")
    ax.legend(fontsize=8)
    ax.grid(False)
    style.frame_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =============================================================================
# Stage entry points
# =============================================================================
def run(cfg, X, y, out_dir):
    out_dir = str(out_dir)
    t, m = cfg["threshold_sweep"], cfg["models"]
    p = lambda name: os.path.join(out_dir, name)        # noqa: E731

    groups = None
    if cfg["data"]["group_column"]:
        groups = load_column(cfg, cfg["data"]["group_column"])
    outputs.start(out_dir, cfg)
    cv_name = "GroupKFold" if groups is not None else "KFold(shuffle)"
    n_feat = X.shape[1]
    print(f"{len(X)} rows, {n_feat} descriptors | CV = {cv_name} "
          f"({m['cv_folds']} folds)")

    X_train, y_train = selection_split(X, y, cfg)
    best_rf_cv = _rf_scorer(X, y, groups, cfg)
    selection_params = models.xgb_params(cfg)

    # ---- 1. XGBoost as a model, selection-stage settings --------------------
    r2_orig, sd_orig = xgb_cv_r2(selection_params, X, y, groups, cfg)
    print(f"[1] XGBoost model, selection settings: CV R\u00b2 = "
          f"{r2_orig:.4f} +/- {sd_orig:.4f}")

    # ---- 2. tuned XGBoost ---------------------------------------------------
    if t["search_iter"] > 0:
        best_params, search_r2 = tune_xgb(X, y, groups, cfg)
        best_params = {k: _plain(v) for k, v in best_params.items()}
        tuned_params = {**selection_params, **best_params}
        r2_tuned, sd_tuned = xgb_cv_r2(tuned_params, X, y, groups, cfg)
        print(f"[2] search ({t['search_iter']} settings, {t['search_folds']}"
              f"-fold) best R\u00b2 = {search_r2:.4f}; tuned model CV R\u00b2 = "
              f"{r2_tuned:.4f} +/- {sd_tuned:.4f}")
    else:
        best_params, search_r2 = {}, np.nan
        tuned_params = dict(selection_params)
        r2_tuned, sd_tuned = r2_orig, sd_orig
        print("[2] tuning skipped (search_iter = 0); ranking with the "
              "selection settings")
    with open(p("xgb_tuned_params.json"), "w") as fh:
        json.dump({"search_iter": t["search_iter"],
                   "search_folds": t["search_folds"],
                   "search_r2": None if np.isnan(search_r2)
                   else float(search_r2),
                   "best_params": best_params,
                   "tuned_params": {k: _plain(v)
                                    for k, v in tuned_params.items()}},
                  fh, indent=1)

    # ---- rank the descriptors with the tuned model on the selection split ---
    ranker = XGBRegressor(**tuned_params).fit(X_train, y_train)
    gain_imp = pd.Series(ranker.feature_importances_, index=X_train.columns)
    shap_imp, _ = tree_shap(ranker, X_train, cfg)
    ranking = pd.DataFrame({
        "gain": gain_imp,
        "gain_rank": gain_imp.rank(ascending=False, method="min"),
        "shap": shap_imp,
        "shap_rank": shap_imp.rank(ascending=False, method="min"),
    }).sort_values("shap", ascending=False)
    ranking.to_csv(p("xgb_importance_ranking.csv"))
    gain_order = gain_imp.sort_values(ascending=False).index.tolist()
    shap_order = shap_imp.sort_values(ascending=False).index.tolist()

    # ---- 3. feature-count sweep ---------------------------------------------
    print("[3] downstream RF R\u00b2 against the number of top descriptors kept")
    ks = sorted({min(k, n_feat) for k in t["k_values"]} | {n_feat})
    rows = []
    for k in ks:
        r2_gain, mf_gain = best_rf_cv(gain_order[:k])
        r2_shap, mf_shap = best_rf_cv(shap_order[:k])
        rows.append({"k": k, "r2_gain_ranked": r2_gain,
                     "best_mf_gain": mf_gain, "r2_shap_ranked": r2_shap,
                     "best_mf_shap": mf_shap})
        print(f"    k={k:2d} | gain-ranked R\u00b2={r2_gain:.4f} | "
              f"SHAP-ranked R\u00b2={r2_shap:.4f}")
    sweep = pd.DataFrame(rows)
    sweep.to_csv(p("xgb_feature_count_sweep.csv"), index=False)
    r2_all, mf_all = best_rf_cv(list(X.columns))

    # ---- the cut a gain threshold gives with the selection-stage XGBoost ----
    cut_k, r2_cut = np.nan, np.nan
    spec = t["mark_cut"]
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        cut_k = int(spec)
        r2_cut = best_rf_cv(gain_order[:cut_k])[0]
    elif spec is not None:
        sel = models.xgb_selector(cfg).fit(X_train, y_train)
        native = pd.Series(sel.feature_importances_, index=X_train.columns)
        cut_feats = native[native >= _gain_threshold(native, spec)].index
        cut_k = len(cut_feats)
        r2_cut = best_rf_cv(list(cut_feats))[0]

    # ---- 4. threshold comparison --------------------------------------------
    print("[4] SelectFromModel thresholds and fixed top-k cuts")
    thr_rows = []
    for thr in t["thresholds"]:
        sfm = SelectFromModel(XGBRegressor(**tuned_params), threshold=thr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sfm.fit(X_train, y_train)
        feats = X_train.columns[sfm.get_support()].tolist()
        r2, _ = best_rf_cv(feats)
        thr_rows.append({"selector": f"threshold={thr}",
                         "n_selected": len(feats), "downstream_rf_r2": r2})
        print(f"    threshold={str(thr):11s} -> {len(feats):2d} descriptors "
              f"| RF R\u00b2={r2:.4f}")
    for k in t["top_k_values"]:
        if k <= n_feat:
            r2, _ = best_rf_cv(gain_order[:k])
            thr_rows.append({"selector": f"top-{k} (gain)", "n_selected": k,
                             "downstream_rf_r2": r2})
            print(f"    top-{k:<2d} (gain)        -> {k:2d} descriptors | "
                  f"RF R\u00b2={r2:.4f}")
    pd.DataFrame(thr_rows).to_csv(p("xgb_threshold_comparison.csv"),
                                  index=False)

    # ---- summary ------------------------------------------------------------
    best = sweep.loc[sweep[["r2_gain_ranked", "r2_shap_ranked"]]
                     .max(axis=1).idxmax()]
    best_k_r2 = max(best["r2_gain_ranked"], best["r2_shap_ranked"])
    summary = {
        "cv": cv_name, "cv_folds": m["cv_folds"], "n_features": n_feat,
        "xgb_r2_mean": r2_orig, "xgb_r2_std": sd_orig,
        "search_r2": search_r2, "tuned_r2_mean": r2_tuned,
        "tuned_r2_std": sd_tuned,
        "rf_all_r2": r2_all, "rf_all_best_max_features": mf_all,
        "cut_threshold": None if isinstance(spec, (int, np.integer))
        else spec,
        "cut_k": cut_k, "rf_r2_at_cut": r2_cut,
        "best_k": int(best["k"]), "best_k_r2": best_k_r2,
    }
    pd.DataFrame([summary]).to_csv(p("xgb_model_summary.csv"), index=False)

    print(f"\n  XGBoost model R\u00b2 {r2_orig:.3f} (selection settings) / "
          f"{r2_tuned:.3f} (tuned) on all descriptors")
    if not np.isnan(cut_k):
        print(f"  cut at {int(cut_k)} descriptors ({spec}): "
              f"downstream RF R\u00b2 {r2_cut:.3f}")
    print(f"  best in the sweep: R\u00b2 {best_k_r2:.3f} at k = "
          f"{int(best['k'])}; RF on all descriptors: {r2_all:.3f}")
    if groups is None:
        print("  (random K-fold; set data.group_column for a grouped, "
              "leakage-aware check)")


def plot(cfg, out_dir):
    out_dir = str(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    sweep = pd.read_csv(os.path.join(out_dir, "xgb_feature_count_sweep.csv"))
    summary = pd.read_csv(os.path.join(out_dir,
                                       "xgb_model_summary.csv")).iloc[0]
    with style.default_style():
        plot_feature_count_sweep(
            sweep, summary, cfg,
            os.path.join(plots_dir, "xgb_feature_count_sweep.png"))
