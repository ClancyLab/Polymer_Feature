"""Run stages in order and keep each run's outputs in one folder.

    runs/<run_name>/run_info.json      data fingerprint and package versions
    runs/<run_name>/<stage>/           tables, plots/, config_used.yaml

Every stage module has run(cfg, X, y, out_dir), which checks its settings,
empties its folder (outputs.start), fits and writes tables, and
plot(cfg, out_dir), which draws figures from those tables only, into a
plots/ folder emptied first. Stages that fit models (USES_MODELS) run with
the tuning stage's settings when tuning applies; the tuning stage is added
to a run when they need it and its results are missing or out of date.
"""

import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__
from . import config as _config
from . import (cascade, consensus, data, families, histograms, mechanism,
               outputs, selection, stability, style, threshold_sweep, tuning)

# Run order. A stage may read the saved outputs of the stages before it.
STAGES = {
    "data": data,
    "histograms": histograms,
    "tuning": tuning,
    "threshold_sweep": threshold_sweep,
    "selection": selection,
    "families": families,
    "consensus": consensus,
    "stability": stability,
    "mechanism": mechanism,
    "cascade": cascade,
}


def _stage_list(stages):
    if not stages:
        return list(STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise KeyError(f"Unknown stage(s) {unknown}; "
                       f"available: {list(STAGES)}")
    return [s for s in STAGES if s in stages or s == "data"]


def _versions():
    out = {"python": platform.python_version(), "fsphysics": __version__}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost",
                "shap", "matplotlib"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _prepare(source, overrides):
    cfg = _config.load_config(source, overrides)
    run_dir = Path(cfg["output_root"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    style.configure(cfg)
    return cfg, run_dir


def _plot_stage(name, cfg, stage_dir):
    """Draw a stage's figures into an emptied plots/ folder, so none is left
    from a plot with other settings."""
    outputs.clear(stage_dir / "plots")
    STAGES[name].plot(cfg, stage_dir)


def run(source, stages=None, overrides=()):
    """Run the given stages (all if None). 'data' always runs first."""
    cfg, run_dir = _prepare(source, overrides)
    X, y = data.load_dataset(cfg)

    info_path = run_dir / "run_info.json"
    sha = data.fingerprint(X, y)
    if info_path.exists():
        old = json.loads(info_path.read_text()).get("data_sha256")
        if old != sha:
            raise ValueError(
                f"The data file differs from the one {run_dir} was made "
                f"with. Use a new run_name, or delete that folder.")
    info = {"data_file": str(cfg["data"]["file"]), "data_sha256": sha,
            "n_rows": int(len(X)), "n_features": int(X.shape[1]),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "versions": _versions()}
    info_path.write_text(json.dumps(info, indent=1))

    names = tuning.with_tuning(_stage_list(stages), STAGES, cfg, run_dir)
    for name in names:
        stage_dir = run_dir / name
        stage_dir.mkdir(exist_ok=True)
        print(f"\n=== {name} ===")
        stage_cfg = tuning.settings_for(STAGES[name], cfg, run_dir)
        STAGES[name].run(stage_cfg, X, y, stage_dir)
        _plot_stage(name, stage_cfg, stage_dir)
    print(f"\nOutputs: {run_dir.resolve()}")
    return run_dir


def plot(source, stages=None, overrides=()):
    """Redraw figures from the saved tables; nothing is refitted. Without
    stages, every stage that has been run."""
    cfg, run_dir = _prepare(source, overrides)
    for name in _stage_list(stages):
        stage_dir = run_dir / name
        if not stage_dir.exists():
            if stages:
                raise FileNotFoundError(f"{stage_dir} not found; run that "
                                        f"stage first.")
            print(f"{name}: not run yet, so no figures")
            continue
        _plot_stage(name, cfg, stage_dir)
    return run_dir
