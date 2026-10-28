"""Load one dataset file, check it, and make the selection split.

The descriptors are the file's columns other than the target, the group ids
and data.exclude (or the data.features list). Text columns become numbers
with data.encode (each value mapped to a number) or data.one_hot (one 0/1
descriptor per value, named "column: value"); encode runs first, so it can
also merge spellings into one label before one_hot. Rows with a missing
descriptor, target or group id stop the run, or are left out with
data.missing: drop_rows. Every stage sees the same rows, numbered from 0.

This is also the 'data' stage: it runs first in every run and stops the run
if the file has problems the analysis cannot handle. It also profiles the
dataset (size, target, groups, rows left out) in checks.txt, with a note
wherever it falls outside the ranges the methods were built for (TYPICAL;
README.txt, "Typical ranges"). Notes never stop a run.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split

from . import outputs

MISSING = ("error", "drop_rows")

# Rough bounds of the datasets the methods were built for (README.txt,
# "Typical ranges"); outside them the data stage adds a note.
TYPICAL = {
    "rows": 500,                    # rows in the file, at least
    "split_rows": 200,              # rows in the selection split, at least
    "split_rows_per_descriptor": 10,
    "descriptors": 100,             # at most
    "target_values": 20,            # distinct target values, at least
    "target_skew": 2.0,             # |skewness|, at most
    "target_fold": 100.0,           # max/min |target| of a one-signed target
    "target_sd": 0.1,               # at least, for threshold_sweep's search
    "outlier_z": 6.0,               # robust z-score of a target value
    "groups_per_fold": 5,           # groups per CV fold, at least
}


def _read_table(cfg):
    d = cfg["data"]
    path = Path(d["file"])
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=d["sheet"]), path
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), path
    raise ValueError(f"Unsupported file type '{path.suffix}' "
                     f"(use .xlsx or .csv).")


def _number(value):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not pd.isna(value))


def _text(value):
    """A value as a descriptor name shows it: 2.0 -> "2"."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _check_settings(d, features, path):
    if d["missing"] not in MISSING:
        raise ValueError(f"data.missing: {d['missing']!r} (use "
                         f"{' or '.join(MISSING)})")
    if (not isinstance(d["encode"], dict)
            or not all(isinstance(m, dict) for m in d["encode"].values())):
        raise ValueError("data.encode: a section {column: {value: number or "
                         "label}}")
    if (not isinstance(d["one_hot"], (list, tuple))
            or len(set(d["one_hot"])) != len(d["one_hot"])):
        raise ValueError("data.one_hot: a list of columns, each once")
    for name in ("encode", "one_hot"):
        absent = [c for c in d[name] if c not in features]
        if absent:
            raise KeyError(f"data.{name}: {absent} not among the descriptor "
                           f"columns of {path.name} (see data.exclude and "
                           f"data.features)")


def _map_values(values, mapping, column):
    """data.encode for one column. A value mapped to null becomes
    missing."""
    lookup = dict(mapping)
    lookup.update({str(k): v for k, v in mapping.items()})

    def find(value):
        if pd.isna(value):
            return None
        if value in lookup:
            return lookup[value]
        return lookup.get(str(value), KeyError)

    out = values.map(find)
    unknown = sorted({_text(v) for v, m in zip(values, out) if m is KeyError})
    if unknown:
        more = " ..." if len(unknown) > 10 else ""
        raise ValueError(f"data.encode.{column}: no entry for "
                         f"{unknown[:10]}{more}")
    return out.map(lambda v: np.nan if v is None else v)


def _one_hot(values, column):
    """One 0/1 column per value, the most frequent first; missing where the
    value is missing."""
    counts = values.value_counts()
    levels = sorted(counts.index, key=lambda v: (-counts[v], _text(v)))
    return pd.DataFrame(
        {f"{column}: {_text(v)}": np.where(values.isna(), np.nan,
                                           (values == v).astype(float))
         for v in levels}, index=values.index)


def _encode(X, d):
    """data.encode, then data.one_hot. Returns the table, a line per encoded
    column, and the number of missing values in each column (after encode,
    before one_hot)."""
    done = []
    for column, mapping in d["encode"].items():
        X[column] = _map_values(X[column], mapping, column)
        known = X[column].dropna()
        if all(_number(v) for v in known):
            X[column] = known.astype(float).reindex(X.index)
            done.append(f"{column}: mapped to numbers")
        elif column not in d["one_hot"]:
            raise ValueError(f"data.encode.{column} maps values to text; map "
                             f"them to numbers, or list the column in "
                             f"data.one_hot too")
    n_missing = {c: int(X[c].isna().sum()) for c in X.columns}
    for column in d["one_hot"]:
        dummies = _one_hot(X[column], column)
        clash = [c for c in dummies.columns if c in X.columns]
        if clash:
            raise ValueError(f"data.one_hot: {clash} already exist")
        at = list(X.columns).index(column)
        X = pd.concat([X.iloc[:, :at], dummies, X.iloc[:, at + 1:]], axis=1)
        done.append(f"{column}: one-hot, {dummies.shape[1]} descriptors")
    return X, done, n_missing


def _load(cfg):
    """The data file as the analysis uses it: (X, y, rows, info). X holds
    the descriptors (encoded), y the target, rows the file's rows that are
    used (all columns, numbered from 0), info what was encoded, missing and
    left out."""
    d = cfg["data"]
    df, path = _read_table(cfg)

    target, exclude = d["target"], list(d["exclude"])
    group = d["group_column"]
    if group == target:
        raise ValueError("data.group_column can't be the target.")
    if group:
        exclude.append(group)          # group ids are never a descriptor
    missing = [c for c in [target] + exclude if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in {path.name}: {missing}")

    if d["features"] == "all":
        features = [c for c in df.columns if c != target and c not in exclude]
    else:
        features = list(d["features"])
        absent = [c for c in features if c not in df.columns]
        if absent:
            raise KeyError(f"Listed features not in {path.name}: {absent}")
        if group in features:
            raise ValueError(f"data.group_column '{group}' is also listed in "
                             f"data.features; group ids are not a "
                             f"descriptor.")

    _check_settings(d, features, path)
    X = df[features].copy()
    X, encoded, n_missing = _encode(X, d)
    non_numeric = [c for c in X.columns
                   if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise TypeError(f"Non-numeric columns {non_numeric}: map them to "
                        f"numbers with data.encode, expand them with "
                        f"data.one_hot, or add them to data.exclude.")
    X = X.astype({c: float for c in X.columns if X[c].dtype == bool})
    X.columns = [str(c) for c in X.columns]
    y = df[target]

    n_missing[target] = int(y.isna().sum())
    if group:
        n_missing[group] = int(df[group].isna().sum())
    left = pd.Series(False, index=df.index)
    if d["missing"] == "drop_rows":
        left = X.isna().any(axis=1) | y.isna()
        if group:
            left |= df[group].isna()
        if left.all():
            raise ValueError(f"Every row of {path.name} has a missing value; "
                             f"exclude the emptiest columns first.")
    info = {"file_rows": len(df), "encoded": encoded,
            "missing": {str(c): n for c, n in sorted(n_missing.items(),
                                                     key=lambda kv: -kv[1])
                        if n},
            "left_out": int(left.sum()),
            "target_left_out": y[left].dropna().astype(float).to_numpy()}
    keep = ~left.to_numpy()
    X = X[keep].reset_index(drop=True)
    y = y[keep].reset_index(drop=True).astype(float)
    rows = df[keep].reset_index(drop=True)
    return X, y, rows, info


def load_dataset(cfg):
    """Return (X, y): the descriptor table and the target, in file order
    (rows left out by data.missing removed, then numbered from 0)."""
    X, y, _, _ = _load(cfg)
    return X, y


def load_column(cfg, column):
    """One column of the data file, for the rows the analysis uses, in the
    same order as X: e.g. group ids, which are never a descriptor."""
    _, _, rows, _ = _load(cfg)
    if column not in rows.columns:
        raise KeyError(f"Column '{column}' not found in {cfg['data']['file']}")
    return rows[column].to_numpy()


def fingerprint(X, y):
    """SHA-256 of a table and its target. Same recipe as data_fingerprint()
    in borutashap_bootstrap.py, so later stages can check they used the
    same rows."""
    h = hashlib.sha256()
    h.update("|".join(map(str, X.columns)).encode())
    h.update(np.ascontiguousarray(X.index.to_numpy(dtype=np.int64)).tobytes())
    h.update(np.ascontiguousarray(X.to_numpy(dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(y, dtype=np.float64)).tobytes())
    return h.hexdigest()


def selection_split(X, y, cfg):
    """Rows the selectors are fitted on. With the defaults this is the
    paper's train_test_split(test_size=0.8, random_state=42)."""
    m = cfg["models"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=1.0 - m["selection_fraction"],
        random_state=m["random_state"])
    return X_train, y_train


def check_dataset(X, y):
    """Return (errors, notes, summary). Errors stop the run; notes don't."""
    errors, notes = [], []
    if y.isna().any():
        errors.append(f"target has {int(y.isna().sum())} missing values")
    elif y.nunique() <= 1:
        errors.append("target has a single value")
    n_missing = X.isna().sum()
    errors += [f"{c}: {n} missing values"
               for c, n in n_missing[n_missing > 0].items()]
    n_unique = X.nunique()
    errors += [f"{c}: constant column" for c in n_unique[n_unique <= 1].index]

    duplicated = X.T.duplicated()
    notes += [f"{c}: identical to an earlier column"
              for c in X.columns[duplicated.to_numpy()]]
    n_dup_rows = int(X.duplicated().sum())
    if n_dup_rows:
        notes.append(f"{n_dup_rows} rows repeat another row's descriptors")
    rho = X.rank().corr().abs().to_numpy()
    cols = list(X.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if rho[i, j] > 0.999 and not (duplicated.iloc[i]
                                          or duplicated.iloc[j]):
                notes.append(f"{cols[i]} and {cols[j]}: |Spearman rho| = 1")

    summary = pd.DataFrame({
        "dtype": X.dtypes.astype(str), "n_missing": n_missing,
        "n_unique": n_unique, "min": X.min(), "max": X.max(),
        "mean": X.mean(), "std": X.std()})
    return errors, notes, summary


def profile(X, y, cfg):
    """The dataset against TYPICAL: (lines for checks.txt, notes)."""
    t, m = TYPICAL, cfg["models"]
    n, p = X.shape
    split = selection_split(X, y, cfg)[0].index
    per = len(split) / p
    lines = [f"selection split: {len(split)} rows (models.selection_fraction "
             f"{m['selection_fraction']}), {per:.1f} per descriptor"]
    notes = []
    if n < t["rows"]:
        notes.append(f"{n} rows: with fewer than about {t['rows']}, the "
                     f"selections are noisy (README.txt, Typical ranges)")
    if len(split) < t["split_rows"] or per < t["split_rows_per_descriptor"]:
        notes.append(f"the selection split holds {len(split)} rows, "
                     f"{per:.1f} per descriptor: raise "
                     f"models.selection_fraction (README.txt, Dataset size)")
    if p > t["descriptors"]:
        notes.append(f"{p} descriptors: BorutaShap and the SHAP interactions "
                     f"slow down and the figures crowd; consider dropping "
                     f"near-duplicates first")
    k = cfg["selection"]["xgb_top_k"]
    if p <= k and cfg["selection"]["xgb_selection"] == "shap_topk":
        notes.append(f"{p} descriptors, no more than selection.xgb_top_k "
                     f"({k}): XGBoost selects them all; lower xgb_top_k")

    sd, skew = float(y.std()), float(y.skew())
    lines.append(f"target: {y.nunique()} distinct values, {y.min():.4g} to "
                 f"{y.max():.4g}, mean {y.mean():.4g}, SD {sd:.4g}, "
                 f"skewness {skew:.2f}")
    if y.nunique() < t["target_values"]:
        notes.append(f"the target has {y.nunique()} distinct values; the "
                     f"methods are for a continuous target (regression)")
    size = y.abs()
    spans = (((y > 0).all() or (y < 0).all())
             and size.max() > t["target_fold"] * size.min())
    if spans:
        notes.append(f"the target spans "
                     f"{np.log10(size.max() / size.min()):.1f} orders of "
                     f"magnitude: consider its logarithm")
    elif abs(skew) > t["target_skew"]:
        notes.append(f"target skewness {skew:.1f}: its tail will dominate "
                     f"R\u00b2 and RMSE; consider a transform")
    mad = 1.4826 * float((y - y.median()).abs().median())
    if mad > 0 and not spans:          # outliers mean little before the log
        z = (y - y.median()).abs() / mad
        far = int((z > t["outlier_z"]).sum())
        if far:
            notes.append(f"{far} target value(s) more than "
                         f"{t['outlier_z']:.0f} robust SDs from the median "
                         f"(the farthest: {y[z.idxmax()]:.4g}); check them")
    if sd < t["target_sd"] and cfg["threshold_sweep"]["search_iter"] > 0:
        notes.append(f"target SD {sd:.3g}: threshold_sweep's XGBoost search "
                     f"space (gamma, reg_alpha) assumes an SD near 1; give "
                     f"the target in smaller units, or scale those ranges")

    group = cfg["data"]["group_column"]
    if group:
        ids = pd.Series(load_column(cfg, group))
        n_all, n_split = ids.nunique(), ids.iloc[split].nunique()
        lines.append(f"groups ({group}): {n_all}, {n / n_all:.1f} rows each "
                     f"on average; {n_split} in the selection split")
        folds = cfg["tuning"]["cv_folds"], m["cv_folds"]
        need = t["groups_per_fold"]
        if n_split < need * folds[0] or n_all < need * folds[1]:
            notes.append(f"few groups for grouped CV ({n_split} in the "
                         f"selection split for {folds[0]} tuning folds, "
                         f"{n_all} in all for {folds[1]} threshold_sweep "
                         f"folds): the fold scores will vary widely")
    return lines, notes


def missing_lines(info, y):
    """checks.txt lines on missing values and the rows left out, and a note
    when the rows left out differ in the target."""
    lines, notes = [], []
    if info["missing"]:
        lines.append("missing values: " + ", ".join(
            f"{c} {n}" for c, n in info["missing"].items()))
    out = info["target_left_out"]
    if len(out) >= 3 and len(y) >= 3:
        p = float(mannwhitneyu(y, out).pvalue)
        lines.append(f"target median: {np.median(y):.4g} in the rows kept, "
                     f"{np.median(out):.4g} in the rows left out "
                     f"(Mann-Whitney p = {p:.2g})")
        if p < 0.05:
            notes.append(f"the rows left out differ in the target (p = "
                         f"{p:.2g}): the rows kept are not a random sample "
                         f"of the file, so the results describe complete "
                         f"records")
    return lines, notes


def run(cfg, X, y, out_dir):
    outputs.start(out_dir, cfg)
    d = cfg["data"]
    _, _, rows, info = _load(cfg)
    errors, notes, summary = check_dataset(X, y)
    group = d["group_column"]
    if group and rows[group].isna().any():
        errors.append(f"data.group_column '{group}' has missing values")
    if info["missing"] and d["missing"] == "error":
        errors.append("to leave out the rows with missing values, set "
                      "data.missing: drop_rows; or fill them in the file")
    summary.to_csv(out_dir / "feature_summary.csv")
    left = (f" of {info['file_rows']}; {info['left_out']} left out for "
            f"missing values" if info["left_out"] else "")
    lines = [f"file: {d['file']}",
             f"rows: {len(X)}{left}",
             f"descriptors: {X.shape[1]}",
             f"target: {d['target']}"]
    lines += [f"encoded {e}" for e in info["encoded"]]
    more, more_notes = missing_lines(info, y)
    lines += more
    notes += more_notes
    if not errors:
        more, profile_notes = profile(X, y, cfg)
        lines += more
        notes += profile_notes
    head = len(lines) + 1
    lines += [""] + [f"ERROR: {e}" for e in errors]
    lines += [f"note: {n}" for n in notes]
    if not errors and not notes:
        lines.append("no problems found")
    (out_dir / "checks.txt").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print(f"{len(X)} rows, {X.shape[1]} descriptors, target "
          f"'{cfg['data']['target']}'")
    for line in lines[4:head - 1] + lines[head:]:
        print(f"  {line}")
    if errors:
        raise ValueError("Data check failed:\n  " + "\n  ".join(errors))


def plot(cfg, out_dir):
    """The data stage has no figures; the histograms stage draws each
    descriptor's distribution."""
