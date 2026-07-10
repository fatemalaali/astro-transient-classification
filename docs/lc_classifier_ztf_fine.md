# Light-Curve Classifier — ZTF Gold, fine taxonomy (companion write-up)

Companion to `lc_classifier_ztf_fine.ipynb`. This document records *what* was done
and *why*, in a form ready to lift into the methodology and results chapters. It
uses UK spelling and Harvard author–year citations; a reference list is at the
end. All figures referenced live in `figures/lc/ztf_fine/` (300-dpi PNG + vector
PDF); all saved models live in `models/lc/ztf_fine/<algorithm>/` and
`models/lc/ztf_fine/hierarchical/`.

The notebook delivers the thesis's **stretch (sub-typing) objective**: it
classifies ZTF *gold* objects into a **fine, 9-class taxonomy** from the same
pre-computed ALeRCE light-curve features (g, r bands) used by the coarse
notebook. It deliberately mirrors `lc_classifier_ztf.ipynb` — same data, same
temporal protocol, same four algorithms, same Optuna budgets — so the coarse and
fine results are directly comparable, and it parallels the "native 14-class"
task of `lc_classifier_plasticc.ipynb` on the simulated side. It adds one new
experiment: an ALeRCE-production-style **hierarchical coarse→fine classifier**,
compared against the flat model on the identical test fold (§5).

---

## 1. Data, taxonomy, and the temporal protocol

The data loading, `oid`-alignment integrity check, dead-feature removal
(≥ 99 % NaN on the training fold only) and the time-ordered 70/15/15 split are
identical to the coarse notebook and are not repeated here (see
`lc_classifier_ztf.md` §1). What is new is the label.

### 1.1 Why the raw fine labels cannot be trained on directly

The gold `fine` column carries 25 raw subtypes, produced by the silver-layer
`map_label` rules. Under the deployment-faithful **temporal split**, however,
several raw classes are unevaluable: the Chen+2020 periodic variables were all
detected early in ZTF, so their first-detection epochs cluster at the start of
the survey and the later folds are starved. Concretely, Cepheid (10 objects),
Mira (20) and Blazar (1) have **zero** validation and test objects; DSCT has 1
test object and no validation; RRL has 4 test objects and no validation. A
25-class task would leave those classes untunable and unscoreable — undefined
rather than merely difficult.

### 1.2 The grouped 9-class taxonomy

The raw labels are therefore grouped into an **ALeRCE-style taxonomy** (cf. the
class families of Sánchez-Sáez et al., 2021): SN subtypes merge into their
spectroscopic families, Blazar folds into AGN, Mira into LPV, and the thin
periodic pulsators into VarOther. The mapping (`FINE_MAP`) is stored verbatim in
every model card so the taxonomy decision travels with the model.

| Grouped class | Raw `fine` labels merged | train | val | test |
|---|---|---|---|---|
| SNIa | SN Ia, SN Ia-91T, SN Ia-91bg, SN Ia-CSM, SN Iax | 3,347 | 1,093 | 1,084 |
| SNII | SN II, SN IIP, SN IIn, SN IIb | 995 | 380 | 312 |
| SNIbc | SN Ib, SN Ic, SN Ibc, SN Ic-BL | 242 | 78 | 90 |
| SLSN | SLSN | 50 | 16 | 22 |
| AGN | AGN/QSO, Blazar | 403 | 117 | 61 |
| CV | CV | 504 | 53 | 68 |
| EB | EB | 1,788 | 10 | 91 |
| LPV | LPV, Mira | 260 | 2 | 14 |
| VarOther | VarOther, RRL, DSCT, Cepheid | 680 | 19 | 28 |

Two properties are enforced as hard assertions in the notebook:

- **Every class is present in every temporal fold** — the reason the grouping
  exists at all.
- **The taxonomy nests cleanly under the coarse classes**: {SNIa, SNII, SNIbc,
  SLSN} → SN, {AGN} → AGN, {CV, EB, LPV, VarOther} → VS. This is what makes the
  hierarchical comparison in §5 well-posed.

The only exclusion is the generic `SN` label (19 objects, 9/5/5 across folds): a
spectroscopically-confirmed supernova whose subtype was never recorded cannot be
placed in a family without guessing, so those objects are dropped and the count
is recorded in each model card (`dropped_labels`).

Because 19 objects are removed, the dead-feature set and the split hash are
**recomputed** rather than inherited; each card stores both the fine notebook's
`split_hash` and the coarse notebook's as `coarse_split_hash`, so the fusion
contract is unambiguous.

### 1.3 Class balance

The fine task is far more imbalanced than the coarse one: SNIa has ~3.3 k
training objects against SLSN's 50 — a ~67× spread (vs ~13× coarse). Macro-F1
remains the primary metric for exactly this reason. The balance figure
(`01_class_balance`, log-scale) also exposes the split-taxonomy interaction that
motivates the §3 tuning caveat: EB has 1,788 training objects but only **10** in
validation, and LPV only **2** — the price of a deployment-faithful temporal
split, reported rather than hidden (the same honesty stance the PLAsTiCC
notebook takes for its rare classes).

---

## 2. Pre-modelling exploratory analysis

Five figures, adapted from the coarse notebook:

- **`01_class_balance`** — 9-class counts per temporal split (log-y).
- **`02_missingness`** — unchanged in kind (mean ≈ 38 % missing); for the fine
  task the class-dependence cuts deeper, since periodic-feature *availability*
  itself separates EB/LPV from CV within the VS family.
- **`03_mutual_information`** — the coarse task's amplitude / colour /
  variability-timescale leaders are joined by the SPM (supernova parametric
  model) shape parameters, as expected now that SN families must be separated by
  light-curve shape.
- **`04_correlation_clustermap`** — unchanged; tree ensembles remain insensitive
  to the deliberate redundancy of the ALeRCE set.
- **`05_headline_separation`** — plotted for a readable 6-class subset (the four
  VS classes plus SNIa and AGN). Period is now the *within-family* discriminant:
  LPV pile up at long periods, EB at short, while the non-periodic CV separates
  on amplitude instead.

---

## 3. Algorithms, tuning, and the thin-class caveat

The four algorithms (LightGBM, XGBoost, Balanced RF, MLP), their justification,
and the by-family preprocessing / imbalance handling are **identical to the
coarse notebook** — see `lc_classifier_ztf.md` §3 for the full rationale and
citations. Two fine-task-specific notes:

- BRF's per-tree balanced bootstrap now downsamples every class toward SLSN's
  50 training objects (~450 rows per tree), a much harsher regime than coarse —
  a visible BRF quality drop is an expected finding, not a bug.
- The MLP's `RandomOverSampler` inflates the training fold to ~9 × 3.3 k rows,
  roughly doubling its tuning cost.

Tuning is Optuna TPE (Akiba et al., 2019) with the same seeds, search spaces and
budgets as the coarse notebook (60 trials for the boosters, 30 for BRF and the
MLP), objective = **macro-F1 on the temporal validation fold**, all 9 labels
passed explicitly with `zero_division=0`. Each tuned configuration is refit on
train + validation and evaluated once on the test fold.

### The thin-class caveat (reported limitation)

LPV (2 val objects) and EB (10) stay **in** the tuning objective — excluding
them would mean the tuner never sees them, and with `zero_division=0` a
configuration that never predicts a thin class pays a consistent 1/9 penalty, so
nothing is undefined. The limitation is *granularity*: an F1 computed from 2
objects takes only a handful of values, so up to ~0.11 of val macro-F1 is
quantisation noise the TPE sampler can chase. The notebook quantifies this
(its §5.3 table: each tuned model's val macro-F1 with and without {LPV, EB}), and
the val→test gap is visible against the §4 results. This is the honest price of the temporal
protocol and is *not* patched by re-stratifying — the temporal split is the
methodological point of the thesis.

---

## 4. Flat 9-class results

The four tuned models are evaluated once on the held-out future test fold. From
the full run (60/60/30/30 Optuna trials), `figures/lc/ztf_fine/test_metrics.csv`
and `latency.csv`:

| Model | Macro-F1 | Balanced acc. | Accuracy | MCC | ROC-AUC | PR-AUC | Log loss | Latency (ms/obj) |
|---|---|---|---|---|---|---|---|---|
| **XGBoost** | **0.705** | 0.726 | 0.823 | 0.703 | 0.955 | 0.759 | 0.549 | 0.023 |
| LightGBM | 0.670 | 0.680 | 0.818 | 0.686 | 0.956 | 0.751 | 0.527 | 0.013 |
| Balanced RF | 0.564 | 0.688 | 0.677 | 0.537 | 0.925 | 0.671 | 1.148 | 0.066 |
| MLP | 0.552 | 0.586 | 0.711 | 0.491 | 0.865 | 0.580 | 3.528 | 0.007 |

**XGBoost wins** the flat 9-class task (macro-F1 0.705), with LightGBM second
(0.670) — the two gradient-boosted trees again lead clearly over Balanced RF
(0.564) and the MLP (0.552), reproducing the coarse-task ordering on a much
harder problem. The absolute macro-F1 sits far below the coarse 0.946, exactly
because the score is now dominated by a rare-class tail rather than the easy
SN/AGN/VS split. As on the coarse task and the PLAsTiCC native task, all four
models predict in well under a tenth of a millisecond per object, so model choice
is driven purely by accuracy.

**Per-class F1 (XGBoost, test):**

| Class | F1 | test support | train support |
|---|---|---|---|
| SNIa | 0.900 | 1,084 | 3,347 |
| LPV | 0.933 | 14 | 260 |
| AGN | 0.887 | 61 | 403 |
| EB | 0.877 | 91 | 1,788 |
| SNII | 0.733 | 312 | 995 |
| CV | 0.737 | 68 | 504 |
| VarOther | 0.556 | 28 | 680 |
| SLSN | 0.364 | 22 | 50 |
| SNIbc | 0.357 | 90 | 242 |

The ranking is decided in the **tail**: SNIbc (stripped-envelope SNe, which sit
between SNIa and SNII in light-curve shape) and SLSN are the hardest, both near
F1 ≈ 0.36, while the dominant and physically-distinctive classes (SNIa, EB, AGN,
and — despite only 14 test objects — the long-period LPV) are recovered well.
The low SNIbc score at a *non-trivial* training support (242) confirms this is a
genuine astrophysical confusion, not merely a data-starvation artefact.

Figures:

- **`07_macroF1_comparison`** — macro-F1 and balanced accuracy per algorithm.
- **`08_confusion_grid`** — row-normalised 9×9 confusion matrices, with white
  separators at the SN | AGN | VS family boundaries. Within-family confusion
  (SNIbc↔SNIa, EB↔VarOther) is astrophysically plausible and cheap; cross-family
  errors are the ones a broker cares about.
- **`09_perclass_f1`** — per-class F1 heatmap (algorithms × classes). The
  ranking is decided in the tail (SLSN, LPV, VarOther, SNIbc); the dominant
  classes are easy for everyone.
- **`10_roc_pr`**, **`11_feature_importance`** — as in the coarse notebook.
- **`12_perclass_f1_vs_trainsize`** — the headline rare-class-tail figure,
  mirroring the PLAsTiCC native task's diagnostic: per-class test F1 against
  training support (log-x). Where F1 rises with support, the tail is
  data-limited rather than model-limited.

---

## 5. Hierarchical coarse→fine comparison

The ALeRCE production classifier is hierarchical (Sánchez-Sáez et al., 2021): a
top-level model chooses the coarse branch, then per-branch models resolve the
subtype. §7 of the notebook rebuilds that architecture on the gold data and
compares it against the flat winner on the identical test fold, combining
probabilities by chaining:

P(fine) = P(coarse) · P(fine | coarse)

Design decisions:

- **Stage-1 (SN/AGN/VS) is retrained inside the fine notebook**, not loaded from
  `models/lc/ztf/`: the fine object set (19 generic-SN dropped) and recomputed
  dead-feature set differ slightly, and the flat-vs-hierarchical comparison is
  only clean on identical data. The coarse notebook's *tuning effort* is still
  reused — its `best_params.json` is loaded directly, at zero extra Optuna cost.
- **Branch models** (SN branch: SNIa/SNII/SNIbc/SLSN; VS branch:
  CV/EB/LPV/VarOther) reuse the flat winner's tuned hyper-parameters; a
  per-branch re-tune would face even thinner validation slices and is noted as
  future work. The AGN branch is a single fine class, so P(fine|AGN) = 1 and no
  model is needed.
- **Winner family only** — the hierarchy uses the §4-winning algorithm; running
  all four would quadruple the section for little methodological gain.
- Branch conditionals are predicted for **all** test objects (not just the ones
  hard-routed to that branch), which is what makes probability chaining exact;
  a hard-routing variant (argmax coarse, then argmax within branch) is reported
  as a reference.

Artefacts: `figures/lc/ztf_fine/hierarchy_comparison.csv` (full metric battery,
flat vs chained), `13_hierarchy_confusion` (side-by-side confusions),
`14_hierarchy_perclass_delta` (per-class F1 dumbbell), and the composite model
under `models/lc/ztf_fine/hierarchical/` with a top-level `hierarchy_card.json`
recording the wiring, the stage-1 coarse test metrics (quantifying how much
coarse error propagates into every fine prediction) and the composite latency.

### Results

Both the flat and hierarchical models use the §4 winner (XGBoost). From
`figures/lc/ztf_fine/hierarchy_comparison.csv`:

| Model | Macro-F1 | Balanced acc. | Accuracy | MCC | ROC-AUC | PR-AUC | Log loss |
|---|---|---|---|---|---|---|---|
| Flat XGBoost | 0.705 | 0.726 | 0.823 | 0.703 | 0.955 | 0.759 | 0.549 |
| Hierarchical XGBoost | 0.706 | 0.699 | 0.840 | 0.723 | 0.962 | 0.776 | 0.492 |

The two are **statistically indistinguishable on macro-F1** (0.705 vs 0.706), but
the hierarchical decomposition is meaningfully better on the calibration- and
accuracy-sensitive metrics: accuracy 0.840 vs 0.823, MCC 0.723 vs 0.703, and log
loss 0.492 vs 0.549. This is the expected signature of the coarse-first
factorisation — the stage-1 model resolves the easy SN/AGN/VS decision with high
confidence (**coarse test macro-F1 0.944**, essentially the coarse notebook's
0.946), so the probability mass is cleaner even where the fine ranking is
unchanged. Probability chaining and hard routing agree to within 0.001 macro-F1
(0.706 vs 0.707), confirming the branches are decisive enough that the softer
combination adds little. The per-class deltas (figure `14`) show the hierarchy
neither rescues the SNIbc/SLSN tail nor harms the strong classes — the coarse
split was never the bottleneck for the hard subtypes.

**Reading for the thesis:** on this dataset the flat and hierarchical light-curve
classifiers are equivalent in headline macro-F1, so the simpler flat model is the
defensible default; the hierarchical variant's advantage is in probability
quality (log loss, MCC), which matters if downstream fusion consumes the
calibrated scores rather than the argmax.

---

## 6. Risks and limitations

- **Thin validation classes.** LPV (2 val objects) and EB (10) make the tuning
  objective partly quantised (§3); the notebook reports the with/without-thin
  numbers rather than re-stratifying.
- **Temporal split × taxonomy interaction.** The variable-star subtypes are
  concentrated in the training fold because their source catalogue (Chen+2020)
  covers early-ZTF detections; test-fold support for LPV (14) and VarOther (28)
  is small, so their per-class scores carry wide implicit error bars.
- **Grouping granularity.** Folding RRL/DSCT/Cepheid into VarOther sacrifices
  pulsator sub-typing that the raw labels would in principle support; with the
  current split those classes are unevaluable anyway, and the mapping is
  recorded in every model card so the decision is revisitable when more data
  arrives (e.g. a rebuilt gold set with a later time cutoff).
- **Missingness as signal.** As on the coarse task, a model can partly learn
  feature *availability* rather than physics; within the VS family this effect
  is plausibly stronger (periodic features exist only for periodic stars).

---

## 7. Model persistence contract

Each of `models/lc/ztf_fine/{lightgbm,xgboost,brf,mlp}/` contains the standard
contract — `model.joblib`, native booster (`model.txt` / `model.json`),
`best_params.json`, `model_card.json` — with the card extended for the fine
task: `taxonomy: "fine_grouped"`, the 9-class order, **`fine_map` and
`parent_map` verbatim**, `dropped_labels`, and both `split_hash` (fine) and
`coarse_split_hash` (lineage pointer to the coarse notebook's split).

The hierarchical composite lives under `models/lc/ztf_fine/hierarchical/` as
three component directories (`stage1_coarse`, `sn_branch`, `vs_branch`), each
with the same per-component contract plus a `role` field, and a top-level
`hierarchy_card.json` that records the wiring and chaining formula so the whole
hierarchy can be loaded from one file.

---

## References

Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M. (2019) 'Optuna: a
next-generation hyperparameter optimization framework', *Proceedings of the 25th
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*,
pp. 2623–2631.

Chen, X. et al. (2020) 'The Zwicky Transient Facility catalog of periodic
variable stars', *The Astrophysical Journal Supplement Series*, 249(1), 18.

Sánchez-Sáez, P. et al. (2021) 'Alert classification for the ALeRCE broker
system: the light curve classifier', *The Astronomical Journal*, 161(3), 141.

*Algorithm and tooling references (LightGBM, XGBoost, Balanced RF, MLP evidence,
Optuna, imbalanced-learn) are listed in* `lc_classifier_ztf.md`.
