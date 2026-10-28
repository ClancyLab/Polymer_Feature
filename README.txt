fsphysics
=========

Maurice Curran, Clancy group, Johns Hopkins University
https://github.com/ClancyLab/Polymer_Feature

Feature selection and interpretation for descriptor datasets, built from
the analysis scripts of the polymer solvation paper. For one numeric target
(regression only), it tunes its models to the dataset, then finds which
descriptors predict the target, how far three selection methods agree, how
stable their choices are, and how the chosen descriptors act on the target.

Requires Python 3.10 or newer. Work that uses fsphysics should cite the
paper it comes from (see Citing).


Install
-------

Get the code from https://github.com/ClancyLab/Polymer_Feature (Code >
Download ZIP, or git clone). Then, from the downloaded folder (the one with
pyproject.toml in it), in the environment you want to use:

    pip install -e .

The install is editable: changes to the code take effect without
reinstalling. Drawing the optional surrogate trees of the mechanism stage
also needs dtreeviz and Graphviz (pip install -e .[trees], and Graphviz
from conda-forge or graphviz.org); nothing else uses them.


Quick start
-----------

1. Put the dataset in one .xlsx or .csv file: one row per sample, one
   column per descriptor, and the target column. Columns that are not
   descriptors (names, IDs) can stay if they are listed in data.exclude.
   Text columns and missing values need settings (see Clean data below).
2. Copy configs/template.yaml, fill in its data section, and delete the
   settings you want at their defaults. configs/polymer_solvation_paper.yaml
   is a filled-in example; run from this folder, it reproduces the paper
   from the dataset in data/ (see data/README.txt).
3. Run it from the folder that the paths in the config are relative to:

    fsphysics run my.yaml                           every stage
    fsphysics run my.yaml --stages selection        one stage (data runs too)
    fsphysics run my.yaml --set models.cv_folds=5   override a setting once
    fsphysics plot my.yaml                          redraw figures, no refit

Each stage writes to runs/<run_name>/<stage>/: CSV tables, a summary file,
the figures in plots/, and config_used.yaml with every setting it used.
The stages that fit models use the settings the tuning stage found for the
dataset; a run adds the tuning stage when its results are missing or out of
date (a few minutes for a dataset the size of the paper's).

From Python:

    import fsphysics

    if __name__ == "__main__":      # needed on Windows (worker processes)
        fsphysics.run("my.yaml", stages=["selection"],
                      overrides=["selection.xgb_top_k=10"])

The config can also be a Python dict shaped like the YAML file.


How it works
------------

All modules are in src/fsphysics/.

config.py     DEFAULTS holds every setting, with the paper's protocol as the
              values, except that tuning is on (the paper's config turns it
              off). A dataset's YAML lists only what differs, and an unknown
              setting name is an error rather than ignored.
              configs/template.yaml lists every setting with a comment.
pipeline.py   Runs the stages in order (STAGES). The data stage always runs
              first; run_info.json fingerprints the data, and a run folder
              refuses a different data file (change run_name instead).
<stage>.py    One module per stage. run(cfg, X, y, out_dir) checks the
              settings, empties the stage folder and writes the tables;
              plot(cfg, out_dir) draws the figures from those tables only.
              A stage folder holds its last run only, so keep edited
              figures elsewhere.
tuning.py     The tuning stage. It also decides when a run needs it
              (with_tuning) and gives the tuned settings to the stages
              marked USES_MODELS (settings_for).
models.py     The RandomForest, XGBoost and LassoCV models every stage
              uses, built from the models settings.
style.py      Descriptor labels, colours and figure styles.
data.py       Reading and checking the data file: text encoding and
              missing values (_encode, _load), the selection split, the
              typical ranges (TYPICAL).
outputs.py    Emptying a stage folder (start) and descriptor file names.
_vendor/      The BorutaShap.py the paper used (MIT license), for the
              selection and stability stages.


Stages
------

In run order. "Needs" is what a stage reads from earlier stages, "Change"
the settings most worth adjusting, and "Code" where the method itself is.

data             Stops on missing values (unless data.missing is
                 drop_rows), constant columns, text columns not encoded, or
                 a single-valued target; notes identical columns, repeated
                 rows, |rho| = 1 pairs, rows left out, and anything outside
                 the typical ranges below (checks.txt).
histograms       Each descriptor's histogram, stacked by class of the
                 target, with a chi-square test and Cramer's V.
                 Needs: nothing. Change: class_edges, custom_bins,
                 bin_range, features. Code: make_bins, make_classes.
tuning           A grid search for the settings of both RandomForests
                 (rf_selector, rf_downstream) and XGBoost, by K-fold CV on
                 the selection split. Each model's best setting replaces its
                 models-section values in threshold_sweep, selection,
                 stability and mechanism, unless the models section scores
                 as well. tuned_params.yaml holds the settings used;
                 tuning_summary.txt notes any best value at the edge of its
                 grid (widen that grid). Needs: nothing. Change: grids,
                 models ([] = no tuning), search, scoring, cv_folds.
                 Code: search, run.
threshold_sweep  XGBoost as a model (with its own random search, the
                 paper's), and the downstream accuracy for each number of
                 top descriptors; the paper's cut of 8 came from this curve.
                 Needs: nothing. Change: search_iter (0 = no search),
                 k_values.
selection        BorutaShap, Lasso and XGBoost each choose descriptors on
                 the selection split; their importances go on one SHAP
                 scale, and each chosen set is scored by a RandomForest with
                 K-fold CV on all rows. Needs: nothing. Change: methods,
                 boruta_trials, xgb_selection, xgb_top_k, top_n.
                 Code: select_borutashap, select_lasso, select_xgboost,
                 run_rf_grid.
families         Groups collinear descriptors: Spearman |rho|, average
                 linkage, a family where the mean |rho| >= corr_threshold.
                 Needs: nothing. Change: corr_threshold, compare_thresholds,
                 labels.family_names, the dendrogram figure settings.
consensus        How far the methods agree, per descriptor and per family
                 (rank correlations, Kendall's W, top-k overlaps, mean-rank
                 consensus). Needs: selection, families. Change: top_ks.
stability        Each selector refitted on n_boot subsamples: selection
                 frequency, rank spread, Nogueira stability index, robust
                 candidates. Needs: selection, families, consensus.
                 Change: n_boot, subsample_frac, n_workers, the robust_*
                 rule. By far the slowest stage (hours at n_boot 200 on the
                 paper's data); finished refits are saved, so a stopped run
                 resumes where it stopped.
mechanism        SHAP interactions, dependence plots with ALE curves,
                 leave-one-family-out, and optional surrogate trees with
                 explicit thresholds. Needs: families, and consensus when
                 key_features is auto. Change: parts, key_features,
                 pair_plots, the *_samples caps.
cascade          A classification tree on classes of the target, grown from
                 each method's selected descriptors (BorutaShap and XGBoost
                 by default), drawn with a histogram at each split and a pie
                 at each leaf. Needs: selection. Change: class_edges,
                 purity_stop, max_depth, the figure settings.
                 Code: grow (the tree), draw (the figure).


Typical ranges
--------------

The methods were developed on the paper's dataset. The bounds below are
rough guides to the datasets the defaults suit (the paper's values in
brackets). The data stage profiles each dataset in data/checks.txt and
adds a note wherever it falls outside them; a note never stops a run.

Rows             500 or more [9,126]. The selectors see the selection split,
                 models.selection_fraction of the rows [0.2, 1,825 rows];
                 keep it at 200 rows or more and at 10 or more rows per
                 descriptor [87], raising selection_fraction if needed.
Descriptors      About 10 to 100 [21]. With 8 or fewer, the top-8 cuts keep
                 every descriptor: lower selection.xgb_top_k, top_n and
                 mechanism.n_key_features. Above 100, BorutaShap and the
                 SHAP interactions slow down and the figures crowd.
Target values    Continuous, with 20 or more distinct values [SMD solvation
                 free energies, -5.6 to -0.6 eV]. A target with a few levels
                 is a classification problem, which the package does not
                 cover.
Target shape     Roughly symmetric, with |skewness| under 2. Give a positive
                 target that spans orders of magnitude (rates, solubilities,
                 viscosities) as its logarithm. Check any value more than 6
                 robust SDs from the median: a few such values dominate R2
                 and RMSE.
Target scale     SD above 0.1. The tree models and the tuning grids ignore
                 the target's scale, but threshold_sweep's XGBoost search
                 space (gamma, reg_alpha) is in target units and was set for
                 an SD near 1: give a narrow target in smaller units (meV
                 rather than eV), or scale those ranges.
Groups           With data.group_column, 5 or more groups per CV fold [351
                 polymers]; with fewer, the fold scores vary widely.
Classes          The histograms' chi-square test wants 5 or more expected
                 rows per bin and class, so about 5 x n_bins x n_classes
                 rows [650 of 9,126]; histograms_summary.txt gives the share
                 below 5 (exp<5). For a small dataset, use fewer classes and
                 lower cascade.min_leaf [50].
Accuracy         Before reading the rankings, check the downstream CV R2 in
                 the selection summary [0.996 or more]. Below about 0.5,
                 they describe a weak model.


Applying it to another dataset
------------------------------

Target. Regression only. data.target_unit and data.target_label set every
label and table heading (an eV target gets meV in the family figures);
data.unit_conversions adds converted RMSE and MAE columns to the selection
summary.

Clean data. Text columns have to become numbers. data.encode maps each
value to a number (several spellings can share one; null makes a value
missing), and data.one_hot gives one 0/1 descriptor per value, named
"column: value". The two combine: encode can first merge spellings into
labels, which one_hot then expands. Missing values stop the run;
data.missing: drop_rows leaves out every row with a missing descriptor,
target or group id instead, and checks.txt then lists the gaps per column
and compares the target of the rows kept with that of the rows left out.
Exclude mostly empty columns first, so that they do not take many rows with
them. The package does not fill gaps: an invented value would enter the
importances and dependence plots as if it had been measured. Constant
columns stop the run; identical columns and |rho| = 1 pairs only get a
note, and the families stage groups them.

Blocks. When the descriptors describe different parts of each sample
(polymer and solvent here; drug and excipient, or host and guest), give
each part a name prefix and list the prefixes in labels.blocks, e.g.
{"Polymer-": "P", "Solvent-": "S"}. Blocks give the short labels
("P: quadrupole"), the sides of the family dendrogram, and the between- and
within-block interaction shares; without blocks those parts are left out.
labels.short_names labels any descriptor; labels.family_names names
families.

Repeated entities. In the paper each polymer appears in every solvent, so
random K-fold splits put the same polymer in training and test folds and
the CV scores come out optimistic: use them to compare methods. Set
data.group_column to an ID column (the polymer name, say), and the tuning
and threshold_sweep stages split by group instead (GroupKFold); the column
is never used as a descriptor. The other stages split rows at random, as
the paper did. The histogram p-values are optimistic for the same reason
(compare Cramer's V instead).

Dataset size. The selectors see models.selection_fraction of the rows (0.2,
the paper's protocol), and the tuning stage tunes on the same rows. For a
few hundred rows, raise it (0.5 or more) and consider models.cv_folds: 5.
For large datasets, models.shap_max_samples and the mechanism.*_samples
settings cap the slow SHAP and ALE steps, and stability.n_boot sets the
stability stage's run time (try 20 for a first look).

Many descriptors. selection.top_n and max_features_main set what the main
figures show (the *_all files keep the full sweep). The histograms stage
draws two figures per descriptor; histograms.features limits them.

Families. families.corr_threshold (0.7) decides what counts as one family.
family_sensitivity.csv shows the families at compare_thresholds, so you can
see whether a conclusion depends on the cut.

Classes. The cascade and histograms stages bin the target into classes.
auto gives tertiles (cascade) and ten equal-size classes (histograms); set
class_edges to values that mean something for the target, as the paper's
config does.

Models. The tuning stage sets rf_selector, rf_downstream and xgb for each
dataset; LassoCV picks its own alpha. To fix the settings instead (to
compare datasets under one protocol, or to rerun without searching), copy
the blocks of tuning/tuned_params.yaml into the models section and set
tuning.models: []. models.rf_selector, rf_downstream, xgb and lasso pass
any scikit-learn or XGBoost parameter through (set one to null to drop it),
and tuning.grids can search any of them, except random_state and n_jobs,
which the package sets, and rf_downstream's max_features, which the
selection stage sweeps. models.random_state seeds the splits and models.


First runs on a new dataset
---------------------------

Run the data stage alone first. It stops with a message if the file has a
problem, and data/checks.txt profiles the dataset and lists what it
noticed:

    fsphysics run my.yaml --stages data

For a quick first pass over every stage, set stability.n_boot to 20 and
selection.boruta_trials to 30 in the YAML, then raise them again for the
runs you keep. After the first run, read tuning/tuning_summary.txt and
widen any grid whose best value sits at its edge. A stage can be rerun on
its own at any time; later stages read the earlier stages' outputs from
the same run folder. Give each dataset or variant its own run_name.

Scores and tuned settings shift a little between versions of scikit-learn
and XGBoost (scikit-learn 1.9, for one, splits GroupKFold folds differently
from 1.7), so keep one environment for all the runs of one study.
run_info.json records the versions each run used.


Changing the code
-----------------

A new setting: add it with its default to DEFAULTS in config.py and to
configs/template.yaml. Sections with free keys (like labels.short_names)
also go in FREE_FORM. Stages read settings from cfg.

A new stage: a module with run(cfg, X, y, out_dir) and plot(cfg, out_dir),
added to STAGES in pipeline.py at its place in the run order. In run(),
check the settings first, then call outputs.start(out_dir, cfg), which
empties the folder and saves config_used.yaml, then write the tables.
plot() draws only from those tables, into out_dir/plots. If the stage fits
the models, set USES_MODELS = True in its module so that it gets the tuned
settings.

A new selection method: a select_<name> function in selection.py that
returns the selected descriptors, a native importance, and the mean |SHAP|
with its standard error (select_lasso is the simplest example). Add it to
METHODS and _select, give it a label and colour in style.METHOD_LABELS and
METHOD_COLORS, and a refit in stability.py if the stability stage should
cover it.

Other models: models.py builds them. BorutaShap wraps models.rf_selector,
so changing that forest changes BorutaShap too. A model the tuning stage
should tune also needs an entry in tuning.MODEL_NAMES and tuning.estimator,
and a default grid in config.py.


Citing
------

fsphysics comes from the analysis in the paper below. Published work that
uses it, or results from it, should cite that paper:

    M. Curran, P. Mastracco, Y. Feng, H. E. Katz and P. Clancy, What Does
    Feature Selection Say About Your Physics? An Interpretability Analysis
    of Polymer Solvation Descriptors, Digital Discovery, submitted.

The paper builds on

    C. Ganley, Y. Feng, H. S. Karmakar, H. E. Katz and P. Clancy, Polym.
    Chem., 2026, 17, 1177.

The stages run published methods. Cite those behind the results you report:

    BorutaShap (selection, stability)
        E. Keany, BorutaShap: A wrapper feature selection method which
        combines the Boruta feature selection algorithm with Shapley values,
        Zenodo, 2020, DOI: 10.5281/zenodo.4247618.
        M. B. Kursa and W. R. Rudnicki, J. Stat. Softw., 2010, 36, 1.
    SHAP values and interactions (selection, stability, mechanism)
        S. M. Lundberg and S.-I. Lee, Adv. Neural Inf. Process. Syst.,
        2017, 30, 4765.
        S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin,
        B. Nair, R. Katz, J. Himmelfarb, N. Bansal and S.-I. Lee, Nat.
        Mach. Intell., 2020, 2, 56.
    XGBoost (tuning, threshold_sweep, selection, stability, mechanism)
        T. Chen and C. Guestrin, in Proceedings of the 22nd ACM SIGKDD
        International Conference on Knowledge Discovery and Data Mining,
        ACM, New York, 2016, pp. 785-794.
    Random forests (every stage that fits models)
        L. Breiman, Mach. Learn., 2001, 45, 5.
    Lasso (selection, stability)
        R. Tibshirani, J. R. Stat. Soc. Ser. B, 1996, 58, 267.
    scikit-learn (the forests, LassoCV, cross-validation, grid search)
        F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion,
        O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg,
        J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot and
        E. Duchesnay, J. Mach. Learn. Res., 2011, 12, 2825.
    Spearman correlation and average linkage (families)
        C. Spearman, Am. J. Psychol., 1904, 15, 72.
        R. R. Sokal and C. D. Michener, Univ. Kans. Sci. Bull., 1958, 38,
        1409.
    Kendall's W, Jaccard overlaps, Borda ranks (consensus)
        M. G. Kendall and B. Babington Smith, Ann. Math. Stat., 1939, 10,
        275.
        P. Jaccard, New Phytol., 1912, 11, 37.
        J.-C. de Borda, Memoire sur les elections au scrutin, Histoire de
        l'Academie Royale des Sciences, Paris, 1781.
    Stability index (stability)
        S. Nogueira, K. Sechidis and G. Brown, J. Mach. Learn. Res., 2018,
        18, 1.
    ALE curves (mechanism)
        D. W. Apley and J. Zhu, J. R. Stat. Soc. Ser. B, 2020, 82, 1059.
    Classification trees (cascade, mechanism surrogates)
        L. Breiman, J. H. Friedman, R. A. Olshen and C. J. Stone,
        Classification and Regression Trees, Wadsworth, Belmont, CA, 1984.


Help
----

Report problems or ask questions in the repository's Issues tab
(https://github.com/ClancyLab/Polymer_Feature/issues). Include the config
and the data stage's checks.txt where you can.


License
-------

MIT; see LICENSE. src/fsphysics/_vendor/borutashap.py is BorutaShap by
E. Keany (https://github.com/Ekeany/Boruta-Shap), also under the MIT
license, which is in src/fsphysics/_vendor/LICENSE_BorutaShap.txt.
