"""Tuning stage: a grid search for each model's settings.

The models section holds the paper's settings, chosen for the polymer
solvation data. For another dataset this stage searches each model's grid
(tuning.grids) with K-fold cross-validation on the selection split, the rows
the selectors are fitted on. The best setting then replaces the models
section's values in every later stage that fits these models
(threshold_sweep, selection, stability, mechanism), if it scores better
than the models section's own settings and tuning.apply is true. The
paper's config sets tuning.models: [] to keep its fixed settings.

  rf_selector    the RandomForest BorutaShap wraps; also the RandomForest
                 the mechanism stage explains
  rf_downstream  the RandomForest that scores each selected set. Its
                 max_features stays out of the search, since the selection
                 stage sweeps it; it uses all descriptors while tuning.
  xgb            XGBoost (threshold_sweep, selection, stability, mechanism)

Lasso needs no grid: LassoCV chooses its own alpha by cross-validation
(models.lasso.cv).

search: grid tries every combination of the listed values, random tries
n_iter of them. Each setting is scored by tuning.scoring, averaged over
tuning.cv_folds folds: shuffled K-fold, or GroupKFold when data.group_column
is set, so that the rows of one group (one polymer in many solvents, say)
never sit in both the training and the test folds. The models section's own
settings are scored on the same folds, for comparison.

The default grids leave each RandomForest's max_features out (rf_selector
keeps the models section's 0.3). max_features sets how often a descriptor
competes with its correlated partners at a split, so it moves importance
between them more than it changes accuracy, and that sharing of credit is
what the families and consensus stages measure.

Outputs:
  tuned_params.yaml         the tuned settings of each model (to freeze them,
                            copy a block into the config's models section
                            and set tuning.models: [])
  tuning_<model>.csv        every setting tried, with its CV score and rank
  tuning_summary.txt        per model: the best setting against the models
                            section's, and any best value at a grid's edge
  plots/tuning_<model>.png  the CV score against each searched setting
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     RandomizedSearchCV, cross_validate)

from . import models, outputs, style
from .data import load_column, selection_split

MODEL_NAMES = ("rf_selector", "rf_downstream", "xgb")
# scorer -> (metric, label, sign that turns the score into the metric)
SCORING = {"r2": ("r2", "R²", 1.0),
           "neg_root_mean_squared_error": ("rmse", "RMSE", -1.0),
           "neg_mean_absolute_error": ("mae", "MAE", -1.0)}
_SET_BY_PACKAGE = {"random_state", "n_jobs"}
# values a setting cannot usefully go beyond, so a best value there is no
# grid edge. min_child_weight: with the squared error, every row weighs 1,
# so any value up to 1 lets a leaf hold a single row.
_LIMITS = {"max_depth": (1, None), "min_samples_leaf": (1, None),
           "min_samples_split": (2, None), "max_features": (None, 1.0),
           "subsample": (None, 1.0), "colsample_bytree": (None, 1.0),
           "colsample_bylevel": (None, 1.0), "colsample_bynode": (None, 1.0),
           "min_child_weight": (1, None), "gamma": (0, None),
           "reg_alpha": (0, None), "reg_lambda": (0, None)}
USES_MODELS = False        # the tuning stage starts from the models section


# =============================================================================
# Settings
# =============================================================================
def _whole(value, low):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and value == int(value) and value >= low)


def _scalar(value):
    return value is None or isinstance(value, (bool, int, float, str))


def estimator(name, cfg):
    """The model as the later stages build it, on one thread (the search
    runs fits in parallel instead)."""
    if name == "rf_selector":
        return models.rf_selector(cfg, n_jobs=1)
    if name == "rf_downstream":
        return models.rf_downstream(cfg, max_features=1.0, n_jobs=1)
    return models.xgb_selector(cfg, n_jobs=1)


def check_settings(cfg):
    """The tuning section, checked; raises with the setting's name."""
    t = cfg["tuning"]
    names = t["models"]
    if (not isinstance(names, (list, tuple)) or len(set(names)) != len(names)
            or any(n not in MODEL_NAMES for n in names)):
        raise ValueError(f"tuning.models: {names!r} (use a list of "
                         f"{', '.join(MODEL_NAMES)}, or [] for no tuning)")
    tests = {
        "apply": (lambda v: isinstance(v, bool), "true or false"),
        "search": (lambda v: v in ("grid", "random"), "grid | random"),
        "n_iter": (lambda v: _whole(v, 1), "a whole number, 1 or more"),
        "cv_folds": (lambda v: _whole(v, 2), "a whole number, 2 or more"),
        "scoring": (lambda v: v in SCORING, " | ".join(SCORING)),
        "n_jobs": (lambda v: v is None or v == -1 or _whole(v, 1),
                   "a whole number, 1 or more, or null for every CPU"),
    }
    for key, (ok, need) in tests.items():
        if not ok(t[key]):
            raise ValueError(f"tuning.{key}: {t[key]!r} (use {need})")
    if not isinstance(t["grids"], dict):
        raise ValueError("tuning.grids: a section {model: {setting: "
                         "[values]}}")
    for name in names:
        grid = t["grids"].get(name)
        where = f"tuning.grids.{name}"
        if not isinstance(grid, dict) or not grid:
            raise ValueError(f"{where}: give the settings to search, e.g. "
                             f"{{max_depth: [6, 10, null]}}")
        known = estimator(name, cfg).get_params()
        for key, values in grid.items():
            if key in _SET_BY_PACKAGE:
                raise ValueError(f"{where}.{key}: the package sets it "
                                 f"(models.random_state, one thread per fit)")
            if name == "rf_downstream" and key == "max_features":
                raise ValueError(f"{where}.max_features: the selection stage "
                                 f"sweeps it, so it can't be tuned here")
            if key not in known:
                raise ValueError(f"{where}.{key}: not a setting of this "
                                 f"model")
            if (not isinstance(values, (list, tuple)) or not values
                    or not all(_scalar(v) for v in values)):
                raise ValueError(f"{where}.{key}: {values!r} (use a list of "
                                 f"values)")
    return t


def enabled(cfg):
    """Whether later stages use tuned settings."""
    return bool(cfg["tuning"]["models"]) and cfg["tuning"]["apply"]


def _plain(value):
    """Tuples as lists, as values read back from YAML."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def depends_on(cfg):
    """Everything the tuned settings depend on: if any of it changes, the
    tuning stage has to run again."""
    t, m = cfg["tuning"], cfg["models"]
    return _plain({
        "models": list(t["models"]),
        "grids": {n: t["grids"].get(n) for n in t["models"]},
        "search": t["search"],
        "n_iter": t["n_iter"] if t["search"] == "random" else None,
        "cv_folds": t["cv_folds"],
        "scoring": t["scoring"],
        "base": {n: m[n] for n in t["models"]},
        "random_state": m["random_state"],
        "selection_fraction": m["selection_fraction"],
        "group_column": cfg["data"]["group_column"],
    })


def _saved(run_dir):
    path = Path(run_dir) / "tuning" / "tuned_params.yaml"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def is_current(cfg, run_dir):
    """Tuned settings exist for the current settings."""
    saved = _saved(run_dir)
    return saved is not None and saved.get("depends_on") == depends_on(cfg)


def with_tuning(names, stages, cfg, run_dir):
    """The stage list, with the tuning stage added (in its place) when a
    stage needs tuned settings that do not exist yet. Checks the tuning
    settings first, when they matter, so a wrong one stops the run before
    any stage runs."""
    if "tuning" in names or any(getattr(stages[n], "USES_MODELS", False)
                                for n in names):
        check_settings(cfg)
    if (not enabled(cfg) or "tuning" in names
            or not any(getattr(stages[n], "USES_MODELS", False)
                       for n in names)
            or is_current(cfg, run_dir)):
        return names
    print("note: the tuning stage runs first; these stages use tuned model "
          "settings, and there are none for the current settings yet.")
    return [n for n in stages if n in names or n == "tuning"]


def settings_for(module, cfg, run_dir):
    """The settings a stage runs with: the tuned models, when the stage fits
    models and tuning applies; otherwise the settings as they are."""
    if not getattr(module, "USES_MODELS", False) or not enabled(cfg):
        return cfg
    saved = _saved(run_dir)
    if saved is None:
        raise FileNotFoundError(
            "tuning is on, but the tuning stage has not run for this run "
            "folder. Run it (fsphysics run CONFIG --stages tuning), or set "
            "tuning.models: [] to use the models section as it is.")
    if saved.get("depends_on") != depends_on(cfg):
        raise ValueError(
            "The tuning settings, or the models settings they start from, "
            "changed since the tuning stage ran. Run it again (fsphysics run "
            "CONFIG --stages tuning).")
    out = copy.deepcopy(cfg)
    for name, params in saved["models"].items():
        out["models"][name] = params
    return out


# =============================================================================
# Search
# =============================================================================
def _cv(cfg, n_splits, groups):
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    return KFold(n_splits=n_splits, shuffle=True,
                 random_state=cfg["models"]["random_state"])


def _groups(cfg, X_train, n_folds):
    """Group ids of the selection split's rows, or None."""
    column = cfg["data"]["group_column"]
    if not column:
        return None
    groups = pd.Series(load_column(cfg, column)).iloc[
        X_train.index.to_numpy()].to_numpy()
    if pd.isna(groups).any():
        raise ValueError(f"data.group_column '{column}' has missing values.")
    n = len(pd.unique(groups))
    if n < n_folds:
        raise ValueError(f"The selection split holds {n} groups of "
                         f"'{column}', fewer than tuning.cv_folds "
                         f"({n_folds}); lower tuning.cv_folds or raise "
                         f"models.selection_fraction.")
    return groups


def _text(value):
    """A setting's value as the tables and figures show it."""
    return "None" if value is None else str(value)


def _grid_size(grid):
    return int(np.prod([len(v) for v in grid.values()]))


def search(name, cfg, X, y, groups, n_jobs):
    """(search object, scores of the models section's settings)."""
    t = cfg["tuning"]
    model = estimator(name, cfg)
    grid = {k: list(v) for k, v in t["grids"][name].items()}
    cv = _cv(cfg, t["cv_folds"], groups)
    common = dict(scoring=t["scoring"], cv=cv, n_jobs=n_jobs, refit=False,
                  error_score="raise")
    if t["search"] == "grid":
        found = GridSearchCV(model, grid, **common)
    else:
        found = RandomizedSearchCV(
            model, grid, n_iter=min(t["n_iter"], _grid_size(grid)),
            random_state=cfg["models"]["random_state"], **common)
    found.fit(X, y, groups=groups)
    base = cross_validate(model, X, y, groups=groups, scoring=t["scoring"],
                          cv=cv, n_jobs=n_jobs,
                          error_score="raise")["test_score"]
    return found, base


def results_table(found, grid, metric, sign):
    res = found.cv_results_
    rows = []
    for i, params in enumerate(res["params"]):
        row = {key: _text(params[key]) for key in grid}
        row[f"cv_{metric}_mean"] = sign * float(res["mean_test_score"][i])
        row[f"cv_{metric}_std"] = float(res["std_test_score"][i])
        row["rank"] = int(res["rank_test_score"][i])
        row["fit_seconds"] = float(res["mean_fit_time"][i])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rank", kind="stable")


def edge_notes(name, grid, best):
    """Best values at the edge of their grid, where a wider grid might do
    better. A null in the grid (no limit, as in max_depth) lies beyond the
    largest number, so the largest number is then no edge."""
    notes = []
    for key, values in grid.items():
        numbers = [v for v in values if isinstance(v, (int, float))
                   and not isinstance(v, bool)]
        value = best[key]
        if len(numbers) < 2 or value not in numbers:
            continue
        low, high = _LIMITS.get(key, (None, None))
        if value == min(numbers) and value != low:
            side = "smallest"
        elif value == max(numbers) and value != high and None not in values:
            side = "largest"
        else:
            continue
        notes.append(f"{name}: the best {key} ({_text(value)}) is the "
                     f"{side} value tried; a wider grid might do better.")
    return notes


def _setting_text(params, keys):
    return ", ".join(f"{k}={_text(params[k])}" for k in keys)


# =============================================================================
# Stage entry points
# =============================================================================
def run(cfg, X, y, out_dir):
    out_dir = Path(out_dir)
    t = check_settings(cfg)
    names = list(t["models"])
    X_train, y_train = selection_split(X, y, cfg)
    groups = _groups(cfg, X_train, t["cv_folds"]) if names else None
    outputs.start(out_dir, cfg)
    if not names:
        text = ("tuning.models is empty, so every stage uses the models "
                "section's settings as they are.")
        (out_dir / "tuning_summary.txt").write_text(text + "\n",
                                                    encoding="utf-8")
        print(text)
        return

    metric, label, sign = SCORING[t["scoring"]]
    unit = cfg["data"]["target_unit"]
    n_jobs = -1 if t["n_jobs"] is None else t["n_jobs"]
    cv_name = ("GroupKFold" if groups is not None else "shuffled K-fold")
    lines = [f"Search: {t['search']}, {t['cv_folds']}-fold CV "
             f"({cv_name}) on the selection split, {len(X_train)} rows; "
             f"score: CV {label}" + (f" ({unit})" if unit and metric != "r2"
                                     else ""),
             "Later stages use the tuned settings." if t["apply"] else
             "tuning.apply is false: later stages use the models section.",
             ""]
    tuned, scores, notes = {}, {}, []
    for name in names:
        grid = {k: list(v) for k, v in t["grids"][name].items()}
        tried = (_grid_size(grid) if t["search"] == "grid"
                 else min(t["n_iter"], _grid_size(grid)))
        print(f"{name}: {tried} settings x {t['cv_folds']} folds ...",
              flush=True)
        found, base = search(name, cfg, X_train, y_train, groups, n_jobs)
        table = results_table(found, grid, metric, sign)
        table.to_csv(out_dir / f"tuning_{name}.csv", index=False)
        best = found.best_params_
        i = found.best_index_
        best_score = float(found.cv_results_["mean_test_score"][i])
        best_mean, best_std = sign * best_score, float(
            found.cv_results_["std_test_score"][i])
        base_mean, base_std = sign * float(np.mean(base)), float(np.std(base))
        # a setting replaces the models section's only if it scores better
        kept = float(np.mean(base)) >= best_score
        tuned[name] = (dict(cfg["models"][name]) if kept
                       else {**cfg["models"][name], **best})
        scores[name] = {"metric": metric, "settings_tried": tried,
                        "best_mean": best_mean, "best_std": best_std,
                        "models_section_mean": base_mean,
                        "models_section_std": base_std,
                        "kept_models_section": kept}
        keys = list(grid)
        current = {k: cfg["models"][name].get(k, "default") for k in keys}
        lines += [f"{name} ({tried} settings)",
                  f"  best: {_setting_text(best, keys)}: CV {label} "
                  f"{best_mean:.4f} +/- {best_std:.4f}",
                  f"  models section: {_setting_text(current, keys)}: CV "
                  f"{label} {base_mean:.4f} +/- {base_std:.4f}"]
        if kept:
            lines.append("  no setting scores better than the models "
                         "section's, so it is kept")
        lines += ["  used: " + ", ".join(f"{k}={_text(v)}"
                                         for k, v in tuned[name].items()),
                  ""]
        notes += edge_notes(name, grid, best)
        print(f"  best {_setting_text(best, keys)}: CV {label} "
              f"{best_mean:.4f} (models section {base_mean:.4f})"
              + ("; models section kept" if kept else ""))
    if notes:
        lines += ["Notes:"] + [f"  {n}" for n in notes]
        for note in notes:
            print(f"  note: {note}")
    (out_dir / "tuning_summary.txt").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    header = ("# The settings the tuning stage chose: the best mean CV score "
              "per model.\n# Later stages use them while tuning.apply is "
              "true. To freeze them, copy\n# a model's block into the "
              "config's models section and set tuning.models: [].\n")
    record = {"models": tuned,
              "grids": {n: t["grids"][n] for n in names},
              "scores": scores, "depends_on": depends_on(cfg)}
    (out_dir / "tuned_params.yaml").write_text(
        header + yaml.safe_dump(_plain(record), sort_keys=False,
                                default_flow_style=None, width=79),
        encoding="utf-8")


def plot(cfg, out_dir):
    out_dir = Path(out_dir)
    path = out_dir / "tuned_params.yaml"
    if not path.exists():
        print("Nothing was tuned, so no figures.")
        return
    with open(path, encoding="utf-8") as fh:
        saved = yaml.safe_load(fh)
    unit = cfg["data"]["target_unit"]
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    drawn = 0
    with style.paper_style():
        for name, grid in saved["grids"].items():
            metric = saved["scores"][name]["metric"]
            label = next(lab for m, lab, _ in SCORING.values()
                         if m == metric)
            table = pd.read_csv(out_dir / f"tuning_{name}.csv",
                                dtype={k: str for k in grid},
                                keep_default_na=False)
            ylabel = f"CV {label}" + (f" ({unit})" if unit and metric != "r2"
                                      else "")
            drawn += draw(grid, table, f"cv_{metric}_mean", ylabel,
                          saved["scores"][name]["models_section_mean"],
                          plots_dir / f"tuning_{name}.png")
    print(f"Figures: {drawn} in {plots_dir}")


def draw(grid, table, column, ylabel, reference, path):
    """One panel per searched setting (one with a single value has none):
    every setting's CV score at its value (grey), the mean per value (black
    line), the best setting (red star) and the models section's score
    (dashed). A star on or below the dashed line means the models section
    was kept. Returns whether a figure was drawn."""
    keys = [k for k, values in grid.items() if len(values) > 1]
    if not keys:
        return False
    fig, axes = plt.subplots(1, len(keys), sharey=True, squeeze=False,
                             figsize=(2.6 * len(keys) + 0.6, 3.2))
    rng = np.random.default_rng(0)
    best = table.loc[table["rank"].idxmin()]
    for ax, key in zip(axes[0], keys):
        labels = [_text(v) for v in grid[key]]
        x = table[key].map({lab: i for i, lab in enumerate(labels)})
        jitter = rng.uniform(-0.12, 0.12, len(table))
        ax.scatter(x + jitter, table[column], s=12, color="0.6",
                   linewidths=0, zorder=2)
        means = table.groupby(x)[column].mean().reindex(range(len(labels)))
        ax.plot(range(len(labels)), means.to_numpy(), color="black", lw=1.4,
                marker="o", ms=5, zorder=3, label="mean per value")
        ax.scatter([labels.index(best[key])], [best[column]], s=70,
                   marker="*", color="#c1121f", zorder=4, label="best")
        ax.axhline(reference, color="0.35", lw=1.0, ls="--", zorder=1,
                   label="models section")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_xlabel(key)
        style.frame_axes(ax)
    axes[0][0].set_ylabel(ylabel)
    fig.tight_layout()
    # above the panels, where it covers no points
    handles, texts = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, texts, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.0), fontsize=9, frameon=False,
               handletextpad=0.3, columnspacing=1.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True
