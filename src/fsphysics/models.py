"""Model settings, defined once and used by every stage.

The parameter values come from the config's 'models' section, whose
defaults are the paper's frozen settings.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor


def rf_selector(cfg, n_jobs=-1):
    """The forest BorutaShap wraps; also used for its unified SHAP values.
    The stability stage runs it with n_jobs=1 inside each worker; a
    forest is identical for any n_jobs at a fixed random_state."""
    m = cfg["models"]
    return RandomForestRegressor(**m["rf_selector"],
                                 random_state=m["random_state"],
                                 n_jobs=n_jobs)


def rf_downstream(cfg, max_features, n_jobs=-1):
    """The forest every selected descriptor set is scored with."""
    m = cfg["models"]
    return RandomForestRegressor(**m["rf_downstream"],
                                 max_features=max_features,
                                 random_state=m["random_state"],
                                 n_jobs=n_jobs)


def xgb_params(cfg, n_jobs=-1):
    """Every XGBRegressor setting of the selection stage, as a dict (the
    threshold sweep starts its tuned model from these)."""
    m = cfg["models"]
    return {**m["xgb"], "random_state": m["random_state"], "n_jobs": n_jobs,
            "importance_type": "gain"}


def xgb_selector(cfg, n_jobs=-1):
    return XGBRegressor(**xgb_params(cfg, n_jobs))


def lasso_cv(cfg, n_jobs=-1):
    m = cfg["models"]
    return LassoCV(**m["lasso"], random_state=m["random_state"],
                   n_jobs=n_jobs)
