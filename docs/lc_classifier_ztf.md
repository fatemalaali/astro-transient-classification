# Light-Curve Classifier — ZTF Gold (companion write-up)

Companion to `lc_classifier_ztf.ipynb`. This document records *what* was done and
*why*, in a form ready to lift into the methodology and results chapters. It uses
UK spelling and Harvard author–year citations; a reference list is at the end.
All figures referenced live in `figures/lc/ztf/` (300-dpi PNG + vector PDF); all
saved models live in `models/lc/ztf/<algorithm>/`.

The notebook classifies ZTF *gold* objects into the coarse classes **SN / AGN /
VS** from the pre-computed ALeRCE light-curve features (g, r bands). It is the
real-data half of the light-curve branch; `lc_classifier_plasticc.ipynb` is the
simulated half, and the two deliberately share one experimental protocol so the
sim-to-real comparison in RQ3 is like-for-like.

---

## 1. Data and the temporal protocol

The gold stage of `build_dataset.ipynb` emits five tables joined on the ZTF object
id `oid`: the ALeRCE feature matrix (267 features across the g and r bands), the
labels, object metadata, a pre-computed train/validation/test split, and the image
stamps (not used here). The notebook loads the four tabular tables and re-runs the
`oid`-alignment integrity check from the gold EDA as a hard assertion — if the
tables disagreed on `oid`, every downstream join would silently corrupt the labels.

**Temporal integrity is the central methodological commitment.** The split is
ordered by each object's first-detection epoch (`firstmjd`), so the validation and
test folds are strictly *future* relative to training. The consequences, enforced
throughout the notebook, are:

- No shuffled *k*-fold cross-validation anywhere; hyper-parameters are validated
  on the temporal validation fold only.
- After tuning, each model is refit on **train + validation** and evaluated
  **once** on the test fold — the test set is touched exactly once per final model.
- Early stopping (LightGBM, XGBoost, MLP) uses the temporal validation fold, never
  the test fold.

This mirrors deployment, where a broker must classify tomorrow's alerts from a
model trained on everything up to today, and it is the reason SMOTE and other
synthetic-oversampling schemes are avoided (see §3): interpolating new points from
future objects would leak information across the temporal boundary.

**Class balance.** The gold set is imbalanced — SN dominate, AGN are rare (~5 % of
the training fold). Macro-F1 is therefore adopted as the primary metric, since it
weights the rare AGN class equally with SN. The class proportions are roughly
stable across the three temporal folds (Figure `01_class_balance`), so the balance
the models tune against matches the one they are finally judged on.

**Missingness.** Feature missingness is heavy (mean ≈ 38 % across features) and, as
the gold EDA established, *class-dependent* — periodicity and harmonic features are
systematically absent for non-periodic transients. This is genuine signal, not
noise, and is handled differently by model family (§3). Features that are ≥ 99 %
missing on the training fold are dropped as *dead* (decided on train only, then
applied to all folds); the survivors keep their NaNs for the tree models.
Figure `02_missingness` shows the distribution and the most-affected surviving
features. We flag as a **limitation** that a model can partly learn feature
*availability* rather than physics; a missing-indicator ablation is the natural
extension and is noted in the notebook.

---

## 2. Pre-modelling exploratory analysis

Five figures motivate the modelling choices:

- **`01_class_balance`** — coarse class counts per temporal split.
- **`02_missingness`** — the missingness distribution and worst-affected features.
- **`03_mutual_information`** — the top-20 features by mutual information with the
  coarse label. The most informative are variability-timescale, amplitude and
  colour descriptors — physically what separates explosive transients (SN),
  stochastic accretion (AGN) and periodic stars (VS). No single feature dominates,
  which already favours an additive ensemble over many weak features.
- **`04_correlation_clustermap`** — hierarchical clustering of the 40
  highest-variance features by absolute correlation. Clear blocks of
  highly-correlated features appear (the g/r band-pair variants and the harmonic
  families); the ALeRCE feature set is deliberately redundant. Because tree
  ensembles are insensitive to correlated inputs, no decorrelation or PCA step is
  required — a point in favour of trees for this branch, since a linear or
  distance-based model would have needed one.
- **`05_headline_separation`** — class-conditional distributions of three headline
  features (amplitude, g−r colour, multiband period). Period isolates the periodic
  VS from the transients; amplitude and colour separate SN from AGN. The classes
  are separable but not *linearly* separable in any single feature, so the
  interactions a tree captures are what raise accuracy.

---

## 3. Algorithm selection and justification

The gold light-curve branch is *tabular* — each object is already summarised by
engineered features rather than a raw time series — so the candidate pool is
tabular learners, not sequence models. Four algorithms are compared, each with a
specific justification.

**LightGBM — primary candidate.** Gradient-boosted decision trees won PLAsTiCC
outright: Boone (2019) took first place with a LightGBM model on
Gaussian-process-augmented features, and the challenge results were later reported
by Hložek et al. (2023). LightGBM (Ke et al., 2017) also splits on missing values
natively, which is directly relevant given the heavy, class-dependent missingness
in the ALeRCE features. It matches the gradient-boosted-tree branch specified in
the thesis proposal.

**Balanced Random Forest — literature-aligned baseline.** The ALeRCE production
light-curve classifier (Sánchez-Sáez et al., 2021) is a Balanced Random Forest
trained on essentially the same feature family as the gold tabular branch. Using
it makes the notebook directly comparable to the broker the pipeline is modelled
on, and its per-tree balanced bootstrap addresses class imbalance without any
synthetic oversampling. The implementation is that of imbalanced-learn
(Lemaître et al., 2017).

**XGBoost — gradient-boosting robustness check.** A second, independent
gradient-boosting implementation (Chen & Guestrin, 2016). Its inclusion shows the
"trees win" conclusion is not specific to one library — strengthening the
methodology chapter rather than adding a genuinely different model class.

**MLP — neural-tabular baseline.** A tuned multilayer perceptron documents *why*
the thesis does not use deep learning on the tabular branch. Tree ensembles
systematically outperform neural networks on medium-sized tabular data
(Grinsztajn et al., 2022), and even ATAT's transformer beat the ALeRCE random-forest
baseline by only a few macro-F1 points with far more machinery
(Cabrera-Vives et al., 2024). An MLP that loses — or barely matches — at higher
cost is the evidence for that design decision.

**Alternatives considered and rejected.** Sequence models on raw photometry
(SuperNNova, Möller & de Boissière, 2020; RAPID; ATAT, Cabrera-Vives et al., 2024)
were rejected because (a) the gold branch stores pre-computed ALeRCE features, not
raw curves; (b) ~11 k labelled ZTF objects is small for sequence deep learning; and
(c) the fusion design deliberately pairs a *feature-based* light-curve branch with
the CNN image branch, so the tabular model is the intended architectural choice.

### Preprocessing and imbalance handling by model family

| Model | Missing values | Imbalance | Scaling |
|---|---|---|---|
| LightGBM | kept (native NaN split) | `class_weight='balanced'` | none |
| XGBoost | kept (native NaN split) | balanced `sample_weight` | none |
| Balanced RF | median impute + missing-indicator | per-tree balanced bootstrap | none |
| MLP | median impute + missing-indicator | train-only random oversampling | standardised |

Every transformer is fitted on the training fold only. **No SMOTE** is used
anywhere: for the MLP — which supports neither `class_weight` nor `sample_weight` —
balance is achieved with `RandomOverSampler`, which *replicates real, train-only*
minority objects. This is not synthetic oversampling; no new points are created, so
nothing crosses the temporal boundary.

---

## 4. Hyper-parameter tuning

Tuning uses Optuna (Akiba et al., 2019) with a TPE sampler and a fixed seed. The
objective is **macro-F1 on the temporal validation fold** — the thesis's primary
metric — evaluated for 60 trials each for LightGBM and XGBoost and 30 each for
Balanced RF and the MLP. The full search spaces are written out in the notebook
(`suggest_params`) and reproduced here in condensed form:

- **LightGBM / XGBoost** — number of trees (200–1200), learning rate (0.01–0.2,
  log), tree complexity (`num_leaves` / `max_depth`), row and column subsampling,
  minimum child weight/samples, and L1/L2 regularisation.
- **Balanced RF** — number of trees (200–800), maximum depth, maximum features,
  minimum samples per leaf, split criterion.
- **MLP** — depth (1–3 layers) and width (64/128/256), L2 penalty, initial learning
  rate, batch size.

Figure `06_optuna_history` shows the optimisation history per algorithm; the GBT
studies converge within the first ~20–30 trials, so the budget is comfortable. Each
tuned configuration is refit on train + val (the deployed model) and saved, and a
separate train-only model is scored on validation to give the honest selection
metric. The best parameters are dumped to `best_params.json` beside each model.

### 4.1 Selection, out-of-fold predictions and calibration

The whole notebook obeys one protocol, defined once in the shared module
`protocol.py` and imported here (and by the PLAsTiCC twin and the fusion notebook),
so the four notebooks tell a single story rather than each re-implementing the
rules:

- **Selection is on validation, never test** (`protocol.select_winner`). The winner
  is the argmax of the tuned-config **validation** macro-F1 frame
  (`figures/lc/ztf/val_metrics.csv`); the test fold is not consulted for selection.
- **The winner alone gets out-of-fold (OOF) predictions** by forward-chaining inside
  the training fold (`protocol.forward_chain_oof`): the train objects are ordered by
  `firstmjd` and cut into five contiguous, time-ordered blocks; for block *r+1* the
  model is refit on blocks *0..r* at the tuned `n_estimators` (no early stopping
  inside a block) and used to predict block *r+1*. Block 0 is never predicted. This
  yields honest, never-in-sample probabilities for **6 622** training objects.
- **The calibration temperature is fitted on those OOF predictions**
  (`protocol.fit_temperature`), not on validation and not in-sample. For this
  probabilistic branch the log-probabilities serve as logits.

These artefacts are what the fusion notebook consumes — it fits its meta-learner on
the OOF probabilities and scores the deployed model's test probabilities — so fusion
no longer has to reconstruct a clean fitting fold by hand.

---

## 5. Results

The four tuned models are evaluated once on the held-out future test fold. The full
metric table is written to `figures/lc/ztf/test_metrics.csv`; the headline figures
are:

- **`07_macroF1_comparison`** — macro-F1 and balanced accuracy per algorithm.
- **`08_confusion_grid`** — row-normalised confusion matrices (2×2 grid).
- **`09_perclass_f1`** — per-class F1 heatmap (algorithms × classes).
- **`10_roc_pr`** — ROC and precision–recall overlays.
- **`11_feature_importance`** — LightGBM gain vs Balanced-RF permutation importance.
- **`latency.csv`** — per-object inference latency.

### Test-set metrics

From the full run (60/60/30/30 Optuna trials), evaluated once on the future test
fold (`figures/lc/ztf/test_metrics.csv`):

| Model | Macro-F1 | Balanced acc. | Accuracy | MCC | ROC-AUC | PR-AUC | Log loss |
|---|---|---|---|---|---|---|---|
| **LightGBM** | **0.9457** | 0.944 | 0.981 | 0.928 | 0.994 | 0.977 | 0.101 |
| XGBoost | 0.9403 | 0.941 | 0.978 | 0.916 | 0.992 | 0.974 | 0.079 |
| Balanced RF | 0.8942 | 0.913 | 0.965 | 0.865 | 0.988 | 0.954 | 0.191 |
| MLP | 0.8659 | 0.906 | 0.937 | 0.781 | 0.970 | 0.928 | 0.483 |

**LightGBM wins the selection on validation** (macro-F1 0.9346 against XGBoost's
0.9320, Balanced RF 0.8517 and the MLP 0.8129; `val_metrics.csv`), and — read here
only to grade it, not to select it — leads on the test fold as well (0.9457, with
XGBoost a close 0.9403). The two gradient-boosted trees are separated by far less
than the gap down to Balanced RF, and the MLP trails. LightGBM is the model carried
to the late-fusion notebook, together with its forward-chaining OOF probabilities
(macro-F1 0.960 on the 6 622 OOF objects — strong but not saturated, i.e. genuinely
held-out) and its OOF-fitted temperature *T* = 1.67.

### Findings

- On real ZTF light-curve features, **gradient-boosted trees win** the coarse
  SN / AGN / VS task, reproducing the PLAsTiCC-literature ordering. The tuned MLP
  does not close the gap despite the extra machinery — the evidence the thesis uses
  to justify a feature-based rather than deep-sequence light-curve branch.
- The decisive axis is the **rare AGN class**: the algorithms agree closely on SN
  and VS, and the macro-F1 ranking is essentially a ranking on AGN F1. Balanced RF
  attains high AGN recall but lower precision (its balanced bootstrap over-predicts
  the rare class), which is why its macro-F1 trails the boosters despite a strong
  balanced accuracy.
- **Feature-importance agreement** (Figure `11_feature_importance`): LightGBM and
  Balanced RF rest their decisions on largely the same amplitude, colour and
  variability-timescale features flagged by the mutual-information analysis. That
  two very different learners agree is a robustness point.
- **Latency**: all four models predict in well under a millisecond per object, so
  none is a bottleneck for the real-time alert pipeline; model choice is driven
  purely by accuracy.

The winning model proceeds to the late-fusion notebook, loaded purely from its
`models/lc/ztf/<algorithm>/` directory.

---

## 6. Model persistence contract

Each saved model directory contains the artefacts the fusion notebook and the alert
system consume, so the loading contract is fixed here. The **winner** additionally
carries the OOF branch contract (the last four rows):

- `model.joblib` — the fitted estimator (preprocessing pipeline + model) via joblib.
- `model.txt` / `model.json` — the native LightGBM booster / XGBoost model, for
  dependency-light loading in the alert system.
- `best_params.json` — the tuned hyper-parameters.
- `model_card.json` — the feature list **in order**, class order, taxonomy, dataset,
  the canonical `split_id`, `base_provenance` (`train_val_refit`), and — for the
  winner — `is_branch_winner`, the OOF-fitted `temperature`, `oof_provenance`
  (`forward_chain_5block`) and `oof_metrics`, plus train date, package versions and
  the validation and test metrics.
- `oof_proba.npy` + `oof_oids.npy` — forward-chaining OOF probabilities and their
  oids (the fusion meta-learner's fitting fold).
- `temperature.json` — the scalar temperature, fitted on OOF.
- `test_proba.npy` + `test_oids.npy` (and `val_proba.npy` for diagnostics) — the
  deployed model's probabilities and oids (what fusion scores on).

The fusion notebook loads models purely from these directories, keyed off
`is_branch_winner`, and never re-imports this notebook. The canonical `split_id` is
the same order-independent hash written into the gold `MANIFEST.json`, so fusion can
assert every branch trained on the identical partition.

---

## References

Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M. (2019) 'Optuna: a
next-generation hyperparameter optimization framework', *Proceedings of the 25th
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*,
pp. 2623–2631.

Boone, K. (2019) 'Avocado: photometric classification of astronomical transients
with Gaussian process augmentation', *The Astronomical Journal*, 158(6), 257.

Cabrera-Vives, G. et al. (2024) 'ATAT: Astronomical Transformer for time series and
tabular data', *Astronomy & Astrophysics*, 689, A289.

Chen, T. and Guestrin, C. (2016) 'XGBoost: a scalable tree boosting system',
*Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery
and Data Mining*, pp. 785–794.

Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022) 'Why do tree-based models
still outperform deep learning on typical tabular data?', *Advances in Neural
Information Processing Systems*, 35, pp. 507–520.

Hložek, R. et al. (2023) 'Results of the Photometric LSST Astronomical Time-Series
Classification Challenge (PLAsTiCC)', *The Astrophysical Journal Supplement Series*,
267(2), 25.

Ke, G. et al. (2017) 'LightGBM: a highly efficient gradient boosting decision
tree', *Advances in Neural Information Processing Systems*, 30, pp. 3146–3154.

Lemaître, G., Nogueira, F. and Aridas, C.K. (2017) 'Imbalanced-learn: a Python
toolbox to tackle the curse of imbalanced datasets in machine learning', *Journal
of Machine Learning Research*, 18(17), pp. 1–5.

Möller, A. and de Boissière, T. (2020) 'SuperNNova: an open-source framework for
Bayesian, neural network-based supernova classification', *Monthly Notices of the
Royal Astronomical Society*, 491(3), pp. 4277–4293.

Sánchez-Sáez, P. et al. (2021) 'Alert classification for the ALeRCE broker system:
the light curve classifier', *The Astronomical Journal*, 161(3), 141.


---

## Addendum — protocol alignment (root-level refactor)

Two issues that earlier versions of this branch carried, and that the fusion branch had to
work around, are now **resolved at source** by the shared `protocol.py` (see §4.1).

**Selection is on validation.** The winner is chosen by `protocol.select_winner` on the
tuned-config validation frame (`val_metrics.csv`); `verdict.csv` reports the test grade of
that already-chosen winner and no longer drives the choice. `grep` for a test-based
`idxmax` selection returns nothing in this notebook. (LightGBM wins on validation as it did
on test, so the carried model is unchanged; the LightGBM–XGBoost validation margin, 0.9346
against 0.9320, is small but the selection rule is now the correct one.)

**Fusion no longer consumes memorised validation predictions.** The deployed model is still
the train+val refit — orthodox for deployment — but the fusion meta-learner and the
calibration temperature are fitted on the **forward-chaining OOF** probabilities emitted
here (§4.1), which are genuinely held out (OOF macro-F1 0.960, not the 1.0000 a booster
scores on its own training fold). The old leakage-audit / train-only-refit workaround in
the fusion notebook is therefore gone; see `docs/fusion_ztf.md`.
