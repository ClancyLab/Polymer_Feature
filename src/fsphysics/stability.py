"""Stability stage: how stable each selector's choices are under resampling.

Ported from borutashap_bootstrap.py and from Task 3b of
feature_families_and_stability.py. Each selector is refitted on n_boot random
subsamples of subsample_frac of the selection split, with the selection
stage's settings. Every method sees the same subsamples, and the model seeds
are the same for every refit, so the spread reflects the data rather than the
seeds.

  * BorutaShap: refitted in parallel worker processes (one per logical CPU by
    default; the forest runs on one thread in each, and a forest is identical
    for any n_jobs at a fixed random_state). Selected = accepted (tentative
    counts as not selected); ranked by the trial-averaged z-scored |SHAP|.
    With early_stop, a refit ends once every descriptor is accepted or
    rejected. BorutaShap's accepted set is the union over trials and a
    rejected descriptor is dropped from the data, so the final sets are fixed
    at that trial; only the trial-averaged importance changes, being averaged
    over the trials actually run.
  * Lasso: selected = non-zero coefficient; ranked by mean |SHAP|.
  * XGBoost: the selection stage's rule (top k by mean |SHAP|, or a gain
    threshold); ranked by mean |SHAP|.

Each finished BorutaShap refit is saved in borutashap_runs/ at once, and that
folder is kept when the stage runs again (the rest of the stage folder is
replaced). Refits already done are skipped, provided they were made from the
same rows with the same settings, so an interrupted run resumes where it
stopped. Records written by borutashap_bootstrap.py have the same format and
can be copied into borutashap_runs/ to reuse them.

From the refits: per-descriptor selection frequency, median rank with its
IQR, top-k frequencies, and the Nogueira et al. (2018) set-stability index.
The master table joins these with the consensus ranks, the families and the
selection stage's sets, and flags robust candidates.

Windows: the BorutaShap refits run in separate processes. The fsphysics
command handles this; a Python script that calls fsphysics.run() must make
the call under  if __name__ == "__main__":

Differences from the original scripts, all deliberate:
  * One stage runs the BorutaShap and the Lasso/XGBoost refits on the same
    subsamples, so the original's cross-script checks (settings, data hash,
    subsample rows) become a check that the selection stage used the same
    rows, plus the resume check on every saved refit.
  * The one-thread settings apply to the worker processes only, not to the
    rest of the run, and Ctrl+C stops the refits at once.
  * The full-split BorutaShap refit is compared with the selection stage's
    borutashap_feature_importance.csv, and the result goes into
    stability_summary.txt as well as the console.
  * The Lasso and XGBoost refit matrices are saved, as BorutaShap's were
    (bootstrap_selection_*.csv, bootstrap_importance_*.csv), and every
    method's Nogueira index goes to stability_index.csv.
  * The robust-candidate rule, the methods compared and the figure's order
    are settings; the defaults are the paper's.
  * stability_summary.txt holds the stability results only; the family and
    consensus results are in those stages' summaries.

run() fits and writes the tables; plot() draws selection_frequency.png from
the bootstrap_stability_<method>.csv tables.
"""

import contextlib
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from . import __version__, models, outputs, style
from ._vendor import borutashap as _bs
from .data import fingerprint, selection_split
from .selection import _gain_threshold

USES_MODELS = True         # runs with the tuned models when tuning applies

METHOD_ORDER = ("borutashap", "lasso", "xgboost")

# Master-table column names per method, as in feature_families_and_stability:
# (selected in the selection stage, bootstrap frequency, median rank, IQR)
_MASTER_COLUMNS = {
    "lasso": ("lasso_selected", "lasso_sel_freq", "lasso_rank_median",
              "lasso_rank_IQR"),
    "xgboost": ("xgboost_selected", "xgb_topk_freq", "xgb_rank_median",
                "xgb_rank_IQR"),
    "borutashap": ("boruta_selected", "boruta_sel_freq",
                   "boruta_rank_median", "boruta_rank_IQR"),
}


# =============================================================================
# Subsamples, identifiers, stability index
# =============================================================================
def subsample_positions(n, n_boot, frac, seed):
    """Row positions of every subsample: one rng.choice per refit, i.e. the
    same draws as both original scripts."""
    rng = np.random.default_rng(seed)
    size = int(round(frac * n))
    return np.stack([rng.choice(n, size=size, replace=False)
                     for _ in range(n_boot)])


def input_key(data_hash, index):
    """Identifies the rows a refit was fitted on (used when resuming)."""
    h = hashlib.sha1(data_hash.encode())
    h.update(np.ascontiguousarray(index.to_numpy(dtype=np.int64)).tobytes())
    return h.hexdigest()


def nogueira_stability(Z):
    """Set-stability index of Nogueira et al. (2018) for a binary matrix Z
    of shape (n_runs, n_features). 1 = perfectly stable selection."""
    M, d = Z.shape
    if M < 2:
        return np.nan
    p_f = Z.mean(axis=0)
    s2_f = M / (M - 1) * p_f * (1 - p_f)
    k_bar = Z.sum(axis=1).mean()
    denom = (k_bar / d) * (1 - k_bar / d)
    if denom <= 0:
        return np.nan
    return float(1.0 - s2_f.mean() / denom)


def _collapse_shap(sv):
    sv = np.abs(np.asarray(sv))
    while sv.ndim > 2:
        sv = sv.mean(axis=-1)
    return sv.mean(axis=0)


def _rank_min(imp_vec, feats):
    return pd.Series(imp_vec, index=feats).rank(
        ascending=False, method="min").values


def stability_table(feats, sel, ranks, top_ks):
    q25, med, q75 = np.percentile(ranks, [25, 50, 75], axis=0)
    tbl = pd.DataFrame({
        "selection_freq": sel.mean(axis=0).round(3),
        "rank_median": med,
        "rank_q25": q25,
        "rank_q75": q75,
    }, index=feats)
    for k in top_ks:
        tbl[f"top{k}_freq"] = (ranks <= k).mean(axis=0).round(3)
    return tbl.sort_values("rank_median")


# =============================================================================
# BorutaShap refits (borutashap_bootstrap.py)
# =============================================================================
class _AllDecided(Exception):
    """Raised inside BorutaShap's trial loop once every descriptor is decided."""


class BorutaShapEarlyStop(_bs.BorutaShap):
    """BorutaShap 1.0.x that can stop once no descriptor is tentative.

    Only test_features() and fit() are extended. Every trial up to the stop
    is computed by the unmodified library code, and after the stop the two
    calls that BorutaShap.fit makes after its loop are run as usual.
    """

    def __init__(self, *args, early_stop=True, **kwargs):
        self.early_stop = early_stop
        self.trials_run = 0
        self.all_decided_at = None
        self.decisions_when_all_decided = None
        super().__init__(*args, **kwargs)

    def test_features(self, iteration):
        super().test_features(iteration)
        self.trials_run = iteration
        if self.all_decided_at is not None:
            return
        accepted = {str(f) for f in self.flatten_list(self.accepted_columns)}
        rejected = {str(f) for f in self.flatten_list(self.rejected_columns)}
        rejected -= accepted
        if len(accepted) + len(rejected) == len(self.all_columns):
            self.all_decided_at = iteration
            self.decisions_when_all_decided = {"accepted": sorted(accepted),
                                               "rejected": sorted(rejected)}
            if self.early_stop:
                raise _AllDecided

    def fit(self, X, y, **kwargs):
        try:
            super().fit(X, y, **kwargs)
        except _AllDecided:
            self.store_feature_importance()
            self.calculate_rejected_accepted_tentative(
                verbose=kwargs.get("verbose", True))
        return self


def _no_progress_bar(iterable, *args, **kwargs):
    return iterable


def _boruta_refit(tag, X_sub, y_sub, rf_params, n_trials, early_stop, key,
                  out_path):
    """Fit one BorutaShap refit and save its record. Runs in a worker
    process."""
    # One progress bar per worker would be unreadable; the main process
    # prints a line per finished refit instead.
    _bs.tqdm = _no_progress_bar
    t0 = time.perf_counter()
    selector = BorutaShapEarlyStop(
        model=RandomForestRegressor(**rf_params),
        importance_measure="shap",
        classification=False,
        early_stop=early_stop,
    )
    # fit random_state left at BorutaShap's default (0), as in the paper run
    selector.fit(X_sub, y_sub, n_trials=n_trials, sample=False,
                 train_or_test="train", verbose=False)

    feats = [str(c) for c in X_sub.columns]
    hist = selector.history_x
    hist.columns = [str(c) for c in hist.columns]
    mean_imp = hist[feats].iloc[1:].mean(axis=0)     # as in results_to_csv

    rec = {
        "tag": tag,
        "input_key": key,
        "settings": {"rf_params": {k: v for k, v in rf_params.items()
                                   if k != "n_jobs"},
                     "n_trials": n_trials, "early_stop": early_stop},
        "accepted": sorted(str(f) for f in selector.accepted),
        "rejected": sorted(str(f) for f in selector.rejected),
        "tentative": sorted(str(f) for f in selector.tentative),
        "trials_run": int(selector.trials_run),
        "stopped_early": bool(selector.trials_run < n_trials),
        "all_decided_at": selector.all_decided_at,
        "decisions_when_all_decided": selector.decisions_when_all_decided,
        "mean_importance": {f: float(mean_imp[f]) for f in feats},
        "minutes": (time.perf_counter() - t0) / 60.0,
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, out_path)
    return rec


def load_record(path, key, settings):
    """A finished refit's record, or None if missing or made with other
    inputs."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    if rec.get("input_key") != key or rec.get("settings") != settings:
        return None
    return rec


def aggregate(runs_dir, feats, n_boot):
    recs = []
    for b in range(n_boot):
        with open(os.path.join(runs_dir, f"boot_{b:03d}.json")) as fh:
            recs.append(json.load(fh))

    idx = pd.RangeIndex(n_boot, name="subsample")
    dec = pd.DataFrame("Tentative", index=idx, columns=feats)
    for b, r in enumerate(recs):
        if r["accepted"]:
            dec.loc[b, r["accepted"]] = "Accepted"
        if r["rejected"]:
            dec.loc[b, r["rejected"]] = "Rejected"
    sel = (dec == "Accepted").astype(int)
    imp = pd.DataFrame([[r["mean_importance"][f] for f in feats]
                        for r in recs], index=idx, columns=feats)
    runs = pd.DataFrame({
        "n_accepted": sel.sum(axis=1),
        "n_rejected": (dec == "Rejected").sum(axis=1),
        "n_tentative": (dec == "Tentative").sum(axis=1),
        "trials_run": [r["trials_run"] for r in recs],
        "all_decided_at": [r["all_decided_at"] for r in recs],
        "stopped_early": [r["stopped_early"] for r in recs],
        "minutes": [round(r["minutes"], 2) for r in recs],
    }, index=idx)
    return sel, dec, imp, runs


def _versions():
    out = {"python": platform.python_version(), "fsphysics": __version__,
           "BorutaShap": "1.0.17 (vendored)"}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "shap",
                "statsmodels"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _borutashap_source():
    """Path and SHA-256 of the BorutaShap code that was actually used."""
    path = os.path.abspath(_bs.__file__)
    with open(path, "rb") as fh:
        return {"file": path, "sha256": hashlib.sha256(fh.read()).hexdigest()}


def report_full_split(runs_dir, feats, reference_csv):
    """Compare the full-split refit with the selection stage's BorutaShap
    run. Returns the report as lines (also printed)."""
    with open(os.path.join(runs_dir, "full_split.json")) as fh:
        rec = json.load(fh)
    acc, rej = set(rec["accepted"]), set(rec["rejected"])
    new_dec = {f: ("Accepted" if f in acc else
                   "Rejected" if f in rej else "Tentative") for f in feats}

    lines = [f"[Full-split check] {rec['trials_run']} trials, "
             f"{len(acc)} accepted, {len(rej)} rejected, "
             f"{len(rec['tentative'])} tentative"]
    if not os.path.exists(reference_csv):
        lines.append(f"  {reference_csv} not found, so the selection stage "
                     f"was not compared.")
    else:
        ref = pd.read_csv(reference_csv)
        ref = ref[ref["Decision"] != "Shadow"].set_index("Features")
        ref.index = ref.index.astype(str)
        diffs = [f for f in feats if ref["Decision"].get(f) != new_dec[f]]
        if diffs:
            lines.append("  DECISIONS DIFFER from the selection stage:")
            for f in diffs:
                lines.append(f"    {f}: selection stage "
                             f"{ref['Decision'].get(f)}, now {new_dec[f]}")
        else:
            lines.append(f"  decisions match the selection stage for all "
                         f"{len(feats)} descriptors")
        gap = max(abs(float(ref.loc[f, "Average Feature Importance"])
                      - rec["mean_importance"][f]) for f in feats)
        lines.append(f"  largest difference in mean importance vs the "
                     f"selection stage: {gap:.2e}")
        if gap > 1e-6:
            lines.append("  (nonzero: settings, package versions or input "
                         "data differ from the selection stage's run)")

    if rec["all_decided_at"] is None:
        lines.append("  a descriptor stayed tentative for all trials, so "
                     "early stopping would not have fired on this split")
    else:
        snap = rec["decisions_when_all_decided"]
        same = (snap["accepted"] == rec["accepted"]
                and snap["rejected"] == rec["rejected"])
        lines.append(f"  every descriptor was decided at trial "
                     f"{rec['all_decided_at']}; decisions at that trial "
                     f"{'equal' if same else 'DIFFER FROM'} the final ones")
    for line in lines:
        print(line)
    return lines


def _hms(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


@contextlib.contextmanager
def _worker_environment():
    """Environment the worker processes start with: one BLAS/OpenMP thread
    each (the parallelism comes from the processes) and a non-interactive
    matplotlib backend. Restored afterwards, so the rest of the run is
    unaffected."""
    values = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
              "MPLBACKEND": "Agg"}
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def borutashap_refits(X_train, y_train, positions, data_hash, cfg, out_dir):
    """BorutaShap on every subsample, in parallel. Writes the bootstrap tables
    and returns (selection matrix, rank matrix, info, report lines)."""
    s, m = cfg["stability"], cfg["models"]
    n_boot = len(positions)
    n_trials = cfg["selection"]["boruta_trials"]
    runs_dir = out_dir / "borutashap_runs"
    runs_dir.mkdir(exist_ok=True)
    feats = [str(c) for c in X_train.columns]
    rf_params = {**m["rf_selector"], "random_state": m["random_state"]}

    jobs = []                                   # (tag, X, y, early_stop)
    if s["full_split_check"]:
        jobs.append(("full_split", X_train, y_train, False))
    for b in range(n_boot):
        jobs.append((f"boot_{b:03d}", X_train.iloc[positions[b]],
                     y_train.iloc[positions[b]], s["early_stop"]))

    todo = []
    for tag, Xs, ys, es in jobs:
        key = input_key(data_hash, Xs.index)
        path = str(runs_dir / f"{tag}.json")
        settings = {"rf_params": rf_params, "n_trials": n_trials,
                    "early_stop": es}
        if load_record(path, key, settings) is None:
            todo.append((tag, Xs, ys, es, key, path))

    n_cpu = s["n_workers"] or os.cpu_count() or 1
    if sys.platform == "win32":
        n_cpu = min(n_cpu, 61)                  # Windows process-pool limit
    n_workers = max(1, min(n_cpu, len(todo)))
    print(f"BorutaShap: {len(jobs) - len(todo)} of {len(jobs)} refits "
          f"already done")
    if todo:
        print(f"Running {len(todo)} BorutaShap refits on {n_workers} "
              f"processes (n_trials={n_trials}, early stop="
              f"{s['early_stop']})", flush=True)

    failed = []
    if todo:
        t0 = time.time()
        done_minutes = []
        worker_params = {**rf_params, "n_jobs": 1}
        ctx = mp.get_context("spawn")           # the same on every platform
        with _worker_environment(), \
                ProcessPoolExecutor(max_workers=n_workers,
                                    mp_context=ctx) as ex:
            futures = {
                ex.submit(_boruta_refit, tag, Xs, ys, worker_params,
                          n_trials, es, key, path): tag
                for tag, Xs, ys, es, key, path in todo
            }
            try:
                for k, fut in enumerate(as_completed(futures), start=1):
                    tag = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:
                        failed.append(tag)
                        print(f"[{k:>4}/{len(todo)}] {tag}: FAILED "
                              f"({exc!r})", flush=True)
                        continue
                    done_minutes.append(rec["minutes"])
                    left = len(todo) - k
                    eta = np.mean(done_minutes) * 60 * left / n_workers
                    print(f"[{k:>4}/{len(todo)}] {tag}: "
                          f"{len(rec['accepted'])} accepted, "
                          f"{len(rec['tentative'])} tentative, "
                          f"{rec['trials_run']} trials, "
                          f"{rec['minutes']:.1f} min | elapsed "
                          f"{_hms(time.time() - t0)}, about {_hms(eta)} left",
                          flush=True)
            except KeyboardInterrupt:
                # Stop at once: cancel the queued refits and end the ones in
                # progress. A record is written to a temporary file and then
                # renamed, so no saved record can be left half-written.
                for proc in list((getattr(ex, "_processes", None)
                                  or {}).values()):
                    proc.terminate()
                ex.shutdown(wait=False, cancel_futures=True)
                print("\nStopped. Finished refits are saved in "
                      f"{runs_dir}; run the stage again to continue.",
                      flush=True)
                raise

    if failed:
        raise RuntimeError(f"{len(failed)} BorutaShap refit(s) failed: "
                           f"{failed}. Run the stage again to retry them; "
                           f"finished refits are kept.")

    # ---- aggregate the subsample refits ----
    sel, dec, imp, runs = aggregate(runs_dir, feats, n_boot)
    p = lambda name: out_dir / name                      # noqa: E731
    sel.to_csv(p("bootstrap_selection_borutashap.csv"))
    dec.to_csv(p("bootstrap_decisions_borutashap.csv"))
    imp.to_csv(p("bootstrap_importance_borutashap.csv"))
    runs.to_csv(p("bootstrap_runs_borutashap.csv"))

    probe = _bs.BorutaShap(model=RandomForestRegressor(),
                           importance_measure="shap", classification=False)
    fit_params = inspect.signature(_bs.BorutaShap.fit).parameters

    def fit_default(name):
        par = fit_params.get(name)
        return None if par is None else par.default

    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": m["random_state"],
        "test_size": 1.0 - m["selection_fraction"],
        "n_boot": n_boot,
        "subsample_frac": s["subsample_frac"],
        "subsample_size": int(positions.shape[1]),
        "n_train_rows": int(len(X_train)),
        "data_sha256": data_hash,
        "features": feats,
        "selected_rule": "accepted (tentative counts as not selected)",
        "borutashap": {
            "importance_measure": "shap",
            "classification": False,
            "percentile": probe.percentile,
            "pvalue": probe.pvalue,
            "n_trials": n_trials,
            "sample": False,
            "train_or_test": "train",
            "normalize": fit_default("normalize"),
            "fit_random_state": fit_default("random_state"),
        },
        "rf_params": rf_params,
        "rf_n_jobs": 1,
        "early_stop": s["early_stop"],
        "n_workers": n_cpu,
        "versions": _versions(),
        "borutashap_source": _borutashap_source(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    with open(p("bootstrap_meta_borutashap.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    Z = sel.to_numpy(dtype=bool)
    print(f"BorutaShap: Nogueira set-stability = {nogueira_stability(Z):.3f}"
          f" ({n_boot} subsamples, frac={s['subsample_frac']})")
    print(f"  accepted per refit: mean {runs['n_accepted'].mean():.2f}, "
          f"range {runs['n_accepted'].min()}-{runs['n_accepted'].max()}; "
          f"refits with a tentative descriptor: "
          f"{int((runs['n_tentative'] > 0).sum())}")
    print(f"  trials per refit: median {runs['trials_run'].median():.0f}, "
          f"max {runs['trials_run'].max()}; "
          f"mean refit time {runs['minutes'].mean():.1f} min")

    report = []
    if s["full_split_check"]:
        report = report_full_split(
            runs_dir, feats,
            str(out_dir.parent / "selection" /
                "borutashap_feature_importance.csv"))

    ranks = np.vstack([_rank_min(v, feats)
                       for v in imp.to_numpy(dtype=float)])
    info = {
        "n_trials": n_trials,
        "early_stop": s["early_stop"],
        "mean_accepted": float(Z.sum(axis=1).mean()),
        "runs_with_tentative": int((dec == "Tentative").any(axis=1).sum()),
    }
    return Z, ranks, info, report


# =============================================================================
# Lasso and XGBoost refits (feature_families_and_stability.py, Task 3b)
# =============================================================================
def lasso_xgboost_refits(X_tr, y_tr, positions, cfg, methods, out_dir,
                         verbose_every=25):
    """Lasso and XGBoost on every subsample, with the selection stage's
    rules. Writes their refit matrices and returns {method: (selection
    matrix, rank matrix)}."""
    feats = list(X_tr.columns)
    d = len(feats)
    n_boot = len(positions)
    n_shap = cfg["models"]["shap_max_samples"]
    s = cfg["selection"]
    use = [m for m in ("lasso", "xgboost") if m in methods]
    sel = {m: np.zeros((n_boot, d), dtype=bool) for m in use}
    imp = {m: np.zeros((n_boot, d)) for m in use}

    for b in range(n_boot):
        idx = positions[b]
        Xb, yb = X_tr.iloc[idx], y_tr.iloc[idx]

        # ---- Lasso (rule: non-zero coefficients) ----
        if "lasso" in use:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xb)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                las = models.lasso_cv(cfg).fit(Xs, yb)
            sel["lasso"][b] = np.abs(las.coef_) > 1e-8
            Xs_s = Xs[:n_shap]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sv = shap.LinearExplainer(las, Xs_s).shap_values(Xs_s)
            imp["lasso"][b] = _collapse_shap(sv)

        # ---- XGBoost (rule: mean-|SHAP| top-k, or a gain threshold) ----
        if "xgboost" in use:
            xgb = models.xgb_selector(cfg).fit(Xb, yb)
            Xb_s = Xb.iloc[:n_shap]
            sv = shap.TreeExplainer(xgb).shap_values(Xb_s,
                                                     check_additivity=False)
            v = _collapse_shap(sv)
            if s["xgb_selection"] == "shap_topk":
                order = np.argsort(-v)
                sel["xgboost"][b, order[:s["xgb_top_k"]]] = True
            else:
                gain = pd.Series(xgb.feature_importances_, index=feats)
                thr = _gain_threshold(gain, s["xgb_selection"])
                sel["xgboost"][b] = (gain >= thr).to_numpy()
            imp["xgboost"][b] = v

        if verbose_every and (b + 1) % verbose_every == 0:
            print(f"    refit {b + 1}/{n_boot}", flush=True)

    out = {}
    index = pd.RangeIndex(n_boot, name="subsample")
    for m in use:
        pd.DataFrame(sel[m].astype(int), index=index, columns=feats).to_csv(
            out_dir / f"bootstrap_selection_{m}.csv")
        pd.DataFrame(imp[m], index=index, columns=feats).to_csv(
            out_dir / f"bootstrap_importance_{m}.csv")
        ranks = np.vstack([_rank_min(v, feats) for v in imp[m]])
        out[m] = (sel[m], ranks)
    return out


# =============================================================================
# Master table and figure
# =============================================================================
def master_table(cfg, methods, stab_tables, inputs):
    """Consensus ranks, families, selection-stage sets and bootstrap
    stability in one table, with the robust-candidate flag."""
    s = cfg["stability"]
    master = pd.read_csv(inputs["consensus"], index_col=0)
    fam = pd.read_csv(inputs["families"], index_col=0)
    sel = pd.read_csv(inputs["selection"], index_col=0)
    master = master.join(fam[["family_id", "family_size", "family_side"]])
    in_sets = [m for m in METHOD_ORDER if m in methods]
    master = master.join(sel[in_sets + ["n_methods_selecting"]].rename(
        columns={m: _MASTER_COLUMNS[m][0] for m in in_sets}))
    for m in [x for x in ("lasso", "xgboost", "borutashap") if x in methods]:
        _, freq_col, med_col, iqr_col = _MASTER_COLUMNS[m]
        master[freq_col] = stab_tables[m]["selection_freq"]
        master[med_col] = stab_tables[m]["rank_median"]
        master[iqr_col] = (stab_tables[m]["rank_q75"]
                           - stab_tables[m]["rank_q25"])
    if "borutashap" in methods:
        bor = pd.read_csv(inputs["borutashap"])
        bor = bor[bor["Decision"] != "Shadow"].set_index("Features")
        bor = bor.rename(columns={
            "Average Feature Importance": "boruta_trial_mean",
            "Standard Deviation Importance": "boruta_trial_std",
            "Decision": "boruta_decision",
        })[["boruta_trial_mean", "boruta_trial_std", "boruta_decision"]]
        master = master.join(bor)

    freq_methods = [m for m in s["robust_freq_methods"] if m in methods]
    robust = ((master["n_methods_selecting"] == len(methods))
              & (master["consensus_mean_rank"] <= s["robust_rank_max"]))
    for m in freq_methods:
        robust = robust & (master[_MASTER_COLUMNS[m][1]]
                           >= s["robust_min_freq"])
    master["robust_candidate"] = robust
    return master.sort_values("consensus_mean_rank"), freq_methods


def _frequency_labels(cfg):
    s = cfg["selection"]
    if s["xgb_selection"] == "shap_topk":
        xgb = f"XGBoost (SHAP top-{s['xgb_top_k']})"
    else:
        xgb = f"XGBoost (gain ≥ {s['xgb_selection']})"
    return {"borutashap": "BorutaShap (accepted)",
            "lasso": "Lasso (non-zero coef)", "xgboost": xgb}


def plot_selection_freq(tables, methods, sort_by, labels, out_path):
    """Descriptors on the X axis; selection frequency on Y, one bar per
    method."""
    methods = [m for m in methods if m in tables]
    # x-axis descriptors built as in the original, so tied descriptors keep
    # their order
    if "lasso" in tables and "xgboost" in tables:
        feats = tables["lasso"].index.union(tables["xgboost"].index)
    else:
        feats = pd.Index(sorted(set().union(*(tables[m].index
                                              for m in methods))))
    if sort_by not in methods:
        sort_by = methods[0]
    df = pd.DataFrame({
        m: tables[m]["selection_freq"].reindex(feats) for m in methods
    }).sort_values(sort_by, ascending=False)
    x = np.arange(len(df))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(df) + 2.5), 6.2))
    for i, m in enumerate(methods):
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, df[m], width=width,
               color=style.METHOD_COLORS[m], edgecolor="black", linewidth=0.6,
               label=labels[m])
    ax.set_xticks(x)
    ax.set_xticklabels([style.short(f) for f in df.index])
    style.rotate_xticklabels(ax, fontsize=11)
    ax.set_ylabel("Selection frequency", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(False)
    ax.set_xlim(-0.7, len(df) - 0.3)
    style.frame_axes(ax, linewidth=1.3)
    ax.tick_params(axis="y", labelsize=11)
    # legend above the axes so it never sits on top of the bars
    ax.legend(fontsize=12, frameon=False, ncol=len(methods),
              loc="lower center", bbox_to_anchor=(0.5, 1.01),
              borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Stage entry points
# =============================================================================
def _inputs(run_dir, methods):
    """Earlier stages' outputs this stage needs, checked before any refit."""
    paths = {
        "split": run_dir / "selection" / "split_info.json",
        "selection": run_dir / "selection" / "selected_feature_sets.csv",
        "families": run_dir / "families" / "feature_families.csv",
        "consensus": run_dir / "consensus" / "consensus_rank_table.csv",
    }
    if "borutashap" in methods:
        paths["borutashap"] = (run_dir / "selection" /
                               "borutashap_feature_importance.csv")
    stage_of = {"split": "selection", "selection": "selection",
                "borutashap": "selection", "families": "families",
                "consensus": "consensus"}
    missing = [st for st in ("selection", "families", "consensus")
               if any(stage_of[k] == st and not v.exists()
                      for k, v in paths.items())]
    if missing:
        raise FileNotFoundError(f"Run the {', '.join(missing)} stage(s) "
                                f"first; the stability stage needs their "
                                f"outputs in {run_dir}.")
    sets = pd.read_csv(paths["selection"], index_col=0)
    absent = [m for m in methods if m not in sets.columns]
    if absent:
        raise ValueError(f"The selection stage did not run {absent}; run it "
                         f"again with the same selection.methods.")
    return paths


def run(cfg, X, y, out_dir):
    out_dir = Path(out_dir)
    s, m = cfg["stability"], cfg["models"]
    methods = [x for x in cfg["selection"]["methods"]]
    unknown = [x for x in methods if x not in METHOD_ORDER]
    if unknown:
        raise KeyError(f"Unknown selection method(s) {unknown}")
    inputs = _inputs(out_dir.parent, methods)

    X_train, y_train = selection_split(X, y, cfg)
    feats = list(X_train.columns)
    data_hash = fingerprint(X_train, y_train)
    split = json.loads(inputs["split"].read_text())
    if split.get("data_sha256") != data_hash:
        raise ValueError("The selection stage used a different selection "
                         "split (data, selection_fraction or random_state "
                         "changed); run the selection stage again first.")
    outputs.start(out_dir, cfg, keep=("borutashap_runs",))   # saved refits

    n_boot = s["n_boot"]
    positions = subsample_positions(len(X_train), n_boot,
                                    s["subsample_frac"], m["random_state"])
    np.save(out_dir / "subsample_rows.npy",
            X_train.index.to_numpy()[positions])
    print(f"{n_boot} subsamples of {positions.shape[1]} rows "
          f"({100 * s['subsample_frac']:.0f}% of the {len(X_train)}-row "
          f"selection split), the same for every method")

    results, bor_info, report = {}, None, []
    if "borutashap" in methods:
        t0 = time.time()
        sel_b, ranks_b, bor_info, report = borutashap_refits(
            X_train, y_train, positions, data_hash, cfg, out_dir)
        results["borutashap"] = (sel_b, ranks_b)
        print(f"  BorutaShap refits: {_hms(time.time() - t0)}")
    if "lasso" in methods or "xgboost" in methods:
        t0 = time.time()
        names = " and ".join(x for x in ("lasso", "xgboost") if x in methods)
        print(f"Refitting {names} on {n_boot} subsamples ...", flush=True)
        results.update(lasso_xgboost_refits(X_train, y_train, positions, cfg,
                                            methods, out_dir))
        print(f"  {names} refits: {_hms(time.time() - t0)}")

    summary, stab_tables, index_rows = [], {}, []
    print("\nSet stability (Nogueira et al. 2018):")
    for method in methods:
        sel, ranks = results[method]
        tbl = stability_table(feats, sel, ranks, s["top_ks"])
        tbl.to_csv(out_dir / f"bootstrap_stability_{method}.csv")
        phi = nogueira_stability(sel)
        stab_tables[method] = tbl
        index_rows.append({"method": method, "nogueira_stability": phi,
                           "n_boot": n_boot,
                           "subsample_frac": s["subsample_frac"],
                           "mean_selected": float(sel.sum(axis=1).mean())})
        why = ""
        if np.isnan(phi):
            why = (" (undefined: needs at least 2 refits)" if n_boot < 2 else
                   " (undefined: every refit selected none or all of the "
                   "descriptors)")
        print(f"  {method}: Nogueira stability = {phi:.3f}{why}")
        summary.append(f"{method}: Nogueira set-stability = {phi:.3f} "
                       f"({n_boot} subsamples, frac={s['subsample_frac']})"
                       f"{why}")
    pd.DataFrame(index_rows).to_csv(out_dir / "stability_index.csv",
                                    index=False)
    if bor_info is not None:
        summary.append(
            f"  borutashap refits: n_trials={bor_info['n_trials']}, "
            f"selected = accepted, mean {bor_info['mean_accepted']:.2f} "
            f"accepted per refit, {bor_info['runs_with_tentative']} of "
            f"{n_boot} refits with a tentative descriptor, early stop="
            f"{bor_info['early_stop']}")

    master, freq_methods = master_table(cfg, methods, stab_tables, inputs)
    master.to_csv(out_dir / "master_feature_table.csv")
    robust = master.index[master["robust_candidate"]].tolist()
    print(f"\nRobust candidates ({len(robust)}): {robust}")
    summary.append("")
    summary.append(f"Robust candidates (all methods select, consensus rank "
                   f"<= {s['robust_rank_max']}, sel freq >= "
                   f"{s['robust_min_freq']}): {robust}")
    summary.append(f"  bootstrap frequency required for: "
                   f"{', '.join(freq_methods) if freq_methods else 'none'}")
    if bor_info is not None:
        summary.append("Note: the BorutaShap refits use the selection "
                       "stage's settings and the same subsamples as the "
                       "other methods. A refit stops once every descriptor "
                       "is accepted or rejected, which leaves its selected "
                       "set unchanged.")
    if report:
        summary += [""] + report

    xgb_rule = (f"SHAP top-{cfg['selection']['xgb_top_k']}"
                if cfg["selection"]["xgb_selection"] == "shap_topk"
                else f"gain >= {cfg['selection']['xgb_selection']}")
    header = (f"Settings: n_boot={n_boot}, subsample_frac="
              f"{s['subsample_frac']} ({positions.shape[1]} of "
              f"{len(X_train)} rows), methods={methods}, xgboost rule="
              f"{xgb_rule}, top_ks={s['top_ks']}")
    with open(out_dir / "stability_summary.txt", "w",
              encoding="utf-8") as fh:
        fh.write(header + "\n\n" + "\n".join(summary) + "\n")


def plot(cfg, out_dir):
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    tables = {}
    for m in cfg["selection"]["methods"]:
        path = out_dir / f"bootstrap_stability_{m}.csv"
        if path.exists():
            tables[m] = pd.read_csv(path, index_col=0)
    if not tables:
        raise FileNotFoundError(f"No bootstrap_stability_*.csv in {out_dir}; "
                                f"run the stability stage first.")
    with style.default_style():
        plot_selection_freq(tables, list(tables),
                            cfg["stability"]["frequency_sort_by"],
                            _frequency_labels(cfg),
                            plots_dir / "selection_frequency.png")
