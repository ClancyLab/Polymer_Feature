"""Settings for a run.

The defaults live here, in Python. They are the paper's protocol, except
that the tuning stage searches the models' settings for each new dataset
(the paper's config turns it off). A dataset's YAML file lists only what
differs from them. load_config() merges the two and rejects unknown setting
names, so a typo fails loudly instead of being silently ignored.
"""

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "run_name": None,             # None -> the data file's name
    "output_root": "runs",
    "data": {
        "file": None,
        "sheet": 0,               # Excel sheet, by name or position
        "target": None,
        "target_label": None,     # axis/legend text; None -> target column
        "target_unit": "",        # e.g. "eV"; used in labels and column names
        "unit_conversions": {},   # e.g. {"kcal_mol": 23.0605}
        "exclude": [],            # columns kept in the file but not used
        "features": "all",        # all other numeric columns, or a list
        "group_column": None,     # column of group ids -> GroupKFold in the
                                  # tuning and threshold_sweep stages
        "encode": {},             # {text column: {value: number or label}}
        "one_hot": [],            # text columns -> one 0/1 column per value
        "missing": "error",       # "error" | "drop_rows": leave out rows
                                  # with a missing descriptor/target/group
    },
    "labels": {
        "blocks": {},             # {"Polymer-": "P"} -> "P: ..." labels
        "short_names": {},        # {column: axis label}
        "family_names": {},       # {member descriptor or family id: name}
    },
    "models": {
        "random_state": 42,
        "selection_fraction": 0.2,    # rows used to fit the selectors
        "cv_folds": 10,
        "shap_max_samples": 1000,
        "rf_selector": {"n_estimators": 100, "max_depth": 15,
                        "min_samples_leaf": 2, "max_features": 0.3,
                        "min_samples_split": 2},
        "rf_downstream": {"n_estimators": 100, "max_depth": 15,
                          "min_samples_leaf": 2, "min_samples_split": 2},
        "xgb": {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
                "subsample": 0.8, "colsample_bytree": 0.8},
        "lasso": {"cv": 5, "max_iter": 100000},
    },
    "histograms": {                     # the part3/part4 histogram scripts
        "features": "all",              # all descriptors, or a list
        "n_bins": 13,                   # per descriptor, open tail bins included
        "bin_range": "dense",           # "dense" (part4) | "full" (part3)
        "lower_q": 0.01,                # dense range: these quantiles;
        "upper_q": 0.99,
        "tail_frac": 0.15,              # a longer tail gets an open bin
        "value_bins": True,             # <= n_bins values: one bin per value
        "custom_bins": {},              # {descriptor: {edges, decimals}}
        "class_edges": "auto",          # target values, or auto: quantiles
        "n_classes": 10,                # auto: this many equal-size classes
        "class_order": "descending",    # first (bottom) class: highest values
        "class_labels": "auto",         # list in class order, or auto
        "file_names": {},               # {descriptor: figure/sheet name}
        "workbook": True,               # also histograms.xlsx
        "figure": {
            "plots": ["counts", "percentages"],
            "figsize": [14, 6],
            "dpi": 300,
            "label_fontsize": 14,
            "tick_rotation": 90,
            "count_label": "Count",
            "percent_label": "Percentage (%)",
            "legend_title": "auto",     # auto: "<target_label> range (<unit>)"
            "legend_loc": "upper left",
            "legend_anchor": [1.02, 1],  # None: inside the axes, at legend_loc
            "legend_reverse": False,    # True: legend in stack order
            "legend_units": True,       # data.target_unit after each class
            "class_colors": "auto",     # list in class order, or auto
            "axis_labels": {},          # {descriptor: x-axis label}
            "save_formats": ["png"],
        },
    },
    "tuning": {                         # grid search for the models' settings
        "models": ["rf_selector", "rf_downstream", "xgb"],   # [] = no tuning
        "apply": True,                  # later stages use the tuned settings
        "search": "grid",               # "grid" | "random" (n_iter settings)
        "n_iter": 20,
        "cv_folds": 5,
        "scoring": "r2",                # or neg_root_mean_squared_error, ...
        "n_jobs": None,                 # parallel fits; None = every CPU
        "grids": {                      # each includes the paper's values
            "rf_selector": {"max_depth": [10, 15, 25, None],
                            "min_samples_leaf": [1, 2, 5]},
            "rf_downstream": {"max_depth": [10, 15, 25, None],
                              "min_samples_leaf": [1, 2, 5]},
            "xgb": {"n_estimators": [300, 500, 1000],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "max_depth": [3, 6, 9],
                    "min_child_weight": [1, 5]},
        },
    },
    "threshold_sweep": {
        "search_iter": 50,            # random XGBoost settings tried; 0 = no tuning
        "search_folds": 5,
        "search_space": {             # name: [distribution, a, b] (scipy.stats)
            "n_estimators": ["randint", 200, 1200],
            "learning_rate": ["loguniform", 0.01, 0.3],
            "max_depth": ["randint", 2, 10],
            "min_child_weight": ["randint", 1, 10],
            "subsample": ["uniform", 0.6, 0.4],         # loc, width: 0.6-1.0
            "colsample_bytree": ["uniform", 0.6, 0.4],
            "gamma": ["uniform", 0.0, 5.0],
            "reg_lambda": ["loguniform", 0.01, 10.0],
            "reg_alpha": ["uniform", 0.0, 2.0],
        },
        "k_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 18, 21],
        "thresholds": ["mean", "0.5*mean", "0.25*mean", "median",
                       "0.5*median"],
        "top_k_values": [3, 5, 8, 10],
        "mark_cut": "mean",           # gain threshold, a fixed k, or None
        "cut_label": None,            # None -> "gain >= mean"
    },
    "selection": {
        "methods": ["borutashap", "lasso", "xgboost"],
        "boruta_trials": 100,
        "xgb_selection": "shap_topk",   # or a gain threshold: "mean", 0.01
        "xgb_top_k": 8,
        "max_features_main": 8,         # x range of the main CV figures
        "max_features_full": None,      # None -> number of descriptors
        "residual_plots": True,
        "importance_scale": "fraction",  # "fraction" | "raw" | "minmax"
        "error_bars": "sem",            # "sem" | "none"
        "errorbar_k": 1.0,
        "top_n": 8,
        "mark_selection": ["borutashap", "lasso"],
        "comparison_top_n": "union",    # "union" | "order_by"
        "comparison_order_by": "borutashap",
    },
    "families": {
        "corr_threshold": 0.7,          # families: mean |Spearman rho| >= this
        "compare_thresholds": [0.6, 0.7, 0.8],
        "dendrogram": {                 # the top-down family dendrogram
            "color_scheme": "single",   # "single" ramp | "blocks": one per block
            "cmap": "turbo",
            "cmap_range": [0.10, 0.90],
            "block_cmaps": [["autumn_r", 0.32, 1.00], ["winter", 0.00, 0.72]],
            "above_cut_color": "#9e9e9e",
            "cut_color": "black",
            "bold": True,
            "label_fontsize": 11,
            "label_outline": 0.5,       # black outline on tick labels; 0 = off
            "figsize": [14, 7.5],
            "branch_labels": [],        # [{text, families, kind, dx, dy}]
            "branch_label_fontsize": 14,
            "block_label_fontsize": 15,
            "branch_label_color": "black",
            "label_gap_frac": 0.45,
            "min_gap_above": 0.12,
        },
    },
    "consensus": {
        "top_ks": [5, 8, 10],           # top-k sets compared between methods
    },
    "stability": {
        "n_boot": 200,                  # refits of each selector
        "subsample_frac": 0.8,          # of the selection split, per refit
        "top_ks": [5, 8, 10],           # top-k frequencies in the tables
        "n_workers": None,              # BorutaShap processes; None = per CPU
        "early_stop": True,             # end a BorutaShap refit once decided
        "full_split_check": True,       # one full-split BorutaShap refit
        "frequency_sort_by": "xgboost",  # x order of selection_frequency.png
        "robust_rank_max": 8,           # robust: consensus mean rank <= this
        "robust_min_freq": 0.8,         # ... and bootstrap frequency >= this
        "robust_freq_methods": ["lasso", "xgboost"],   # ... for these methods
    },
    "mechanism": {
        # parts to run, in this order; add "surrogates" for decision-tree
        # thresholds (drawing the trees needs dtreeviz and Graphviz)
        "parts": ["interactions", "dependence_ale", "dependence_comparison",
                  "lofo"],
        "key_features": "auto",         # list, or auto: top n by consensus
        "n_key_features": 8,
        "pair_plots": "auto",           # [[descriptor, partner], ...] or auto
        "n_pair_plots": 5,              # auto: the strongest pairs
        "partner_report": [],           # descriptors whose partners to list
        "shap_samples": 2000,           # rows for SHAP dependence values
        "interaction_samples": 800,     # rows for SHAP interaction values
        "ale_samples": 3000,            # rows for ALE curves
        "ale_bins": 20,                 # quantile bins for ALE
        "lofo_max_features": 10,        # RF max_features, capped at the count
        "dependence_bold": True,        # bold text on the comparison panels
        "family_unit": "auto",          # auto: meV for eV targets, else target
        "family_fig_names": True,       # family names on ticks, not ids
        "family_fig_titles": False,     # captions carry the titles
        "family_fig_decimals": 0,
        "family_fig_dpi": 600,
        "family_fig_vector": True,      # also a .pdf beside each .png
        "family_fig_height": 5.6,
        "main_effect_sort": True,       # bars largest first
        "main_effect_decompose": False,  # stack within-family on main effect
        "main_effect_grid": False,
        "interaction_triangle": "lower",  # "lower" | "upper"
        "interaction_pair_sum": True,   # cell = F[a,b] + F[b,a]
        "interaction_orderings": ["numeric", "grouped"],
        "interaction_block_divider": True,
        "interaction_box": True,        # frame the triangle figures
        "surrogate_depth_single": 2,
        "surrogate_depth_joint": 3,
        "surrogate_min_leaf_frac": 0.02,
        "joint_tree_features": "auto",  # list, or auto: first 4 key features
        "surrogate_drawings": True,     # dtreeviz SVGs (needs Graphviz)
        "surrogate_viz_sample": 1500,   # points shown in the drawings
    },
    "cascade": {
        "methods": "auto",              # list, or auto: borutashap, xgboost
        "class_edges": "auto",          # target values, or auto: tertiles
        "purity_stop": 0.95,            # a branch stops at this purity
        "max_depth": 5,
        "min_leaf_frac": 0.01,          # minimum leaf: this x rows ...
        "min_leaf": 50,                 # ... but at least this many rows
        "figure": {                     # cascade_by_method.py's CONFIG
            # glyph sizes (inches); the scales multiply them
            "hist_scale": 1.5, "pie_scale": 1.4,
            "hist_w_in": 2.30, "hist_h_in": 1.15, "pie_in": 1.45,
            # canvas and white space
            "col_inches": 1, "row_inches": 2.75,
            "margins": [0.025, 0.99, 0.955, 0.045],  # left, right, top, bottom
            "pad_inches": 0.05,
            "auto_fit": True,           # widen the pitches until nothing touches
            "min_edge_len_in": 0.38,
            "leaf_top_gap_in": 0.10,
            "node_top_gap_in": 0.09,
            "leaf_squeeze": 0.9,        # narrower columns for shallow leaves
            "min_leaf_slot": 0.42,
            "align_leaves_bottom": False,
            # edges
            "edge_style": "elbow",      # "elbow" | "straight"
            "elbow_drop_frac": 0.62,
            "elbow_radius_in": 0.16,
            "edge_lw": 1.4,
            "edge_color": "0.25",
            "arrow_head_scale": 13,
            "arrow_label_mode": "none",  # "none" | "symbol" | "value"
            # histograms
            "hist_mode": "split",       # "split" | "plain" | "stacked"
            "hist_bins": 25,
            "hist_color": "#2b2b2b",
            "hist_color_left": "#8a8a8a",
            "hist_color_right": "#242424",
            "hist_edge_color": "white",
            "hist_edge_lw": 0.35,
            "thresh_color": "#c1121f",
            "thresh_lw": 2.2,
            "hist_headroom": 1.30,
            "show_node_n": True,        # drawn only if fs_node_n > 0
            # text
            "bold": True,
            "font_scale": 1.5,
            "fs_tick": 14, "fs_thresh": 14, "fs_feature": 19,
            "fs_node_n": 0, "fs_cond": 16, "fs_pie_pct": 12,
            "fs_leaf_text": 15,
            "mean_symbol": "auto",      # auto: <E> for energy units, else mean
            "leaf_text_units": True,
            "leaf_text_gap_in": 0.09,
            "leaf_text_leading": 1.10,
            "legend_fontsize": 22, "legend_title_fontsize": 24,
            "pie_pct_min": 15,          # wedge % labels hidden below this
            "legend_loc": "upper right",  # or "upper left", "auto"
            "legend_frame": False,
            "panel_labels": {},         # e.g. {"borutashap": "a"}
            "panel_label_fs": 28,
            "class_colors": "auto",     # list (lowest class first) or auto
            # output
            "save_formats": ["png", "pdf"],
            "dpi": 400,
            "audit": True,              # print anything that still touches
            "audit_pad_in": 0.05,
            "tick_drop_in": None,       # None: from the font sizes
            "label_drop_in": None,
            "text_gap_in": 0.07,
        },
    },
}

# Sections whose keys are free-form: any key is allowed, the user's keys are
# merged into the defaults, and a key set to null is removed.
FREE_FORM = {
    "data.unit_conversions", "data.encode", "labels.blocks",
    "labels.short_names", "labels.family_names", "models.rf_selector",
    "models.rf_downstream", "models.xgb", "models.lasso", "tuning.grids",
    "threshold_sweep.search_space",
    "histograms.custom_bins", "histograms.file_names",
    "histograms.figure.axis_labels", "cascade.figure.panel_labels",
}


# Settings that moved: old name -> new name.
MOVED = {"threshold_sweep.group_column": "data.group_column"}


def _merge(base, user, path=""):
    for key, value in user.items():
        where = f"{path}.{key}" if path else key
        if where in MOVED:
            raise KeyError(f"The setting '{where}' is now '{MOVED[where]}'.")
        if key not in base:
            raise KeyError(f"Unknown setting '{where}'. Check the spelling "
                           f"against configs/template.yaml.")
        if isinstance(base[key], dict) and where not in FREE_FORM:
            if not isinstance(value, dict):
                raise TypeError(f"Setting '{where}' must be a section.")
            _merge(base[key], value, where)
        elif isinstance(base[key], dict):
            merged = {**base[key], **(value or {})}
            # a key set to null is removed, e.g. to drop a default parameter
            base[key] = {k: v for k, v in merged.items() if v is not None}
        else:
            base[key] = value


def _apply_override(cfg, text):
    """Apply one 'section.key=value' override; the value is read as YAML."""
    if "=" not in text:
        raise ValueError(f"--set expects KEY=VALUE, got '{text}'")
    dotted, raw = text.split("=", 1)
    patch = yaml.safe_load(raw)
    for key in reversed(dotted.strip().split(".")):
        patch = {key: patch}
    _merge(cfg, patch)


def load_config(source, overrides=()):
    """Full settings for a run.

    source     path to a YAML file, or a dict with the same structure
    overrides  strings such as "selection.xgb_top_k=10"
    """
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    else:
        user = copy.deepcopy(source)
    cfg = copy.deepcopy(DEFAULTS)
    _merge(cfg, user)
    for text in overrides or ():
        _apply_override(cfg, text)

    if not cfg["data"]["file"] or not cfg["data"]["target"]:
        raise ValueError("data.file and data.target must be set.")
    if cfg["run_name"] is None:
        cfg["run_name"] = Path(cfg["data"]["file"]).stem
    if cfg["data"]["target_label"] is None:
        cfg["data"]["target_label"] = cfg["data"]["target"]
    return cfg


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
