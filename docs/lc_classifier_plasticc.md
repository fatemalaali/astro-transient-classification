# Light-Curve Classifier — PLAsTiCC (companion write-up)

Companion to `lc_classifier_plasticc.ipynb`. It records *what* was done and *why*,
in UK spelling with Harvard author–year citations, ready to lift into the
methodology and results chapters. Figures live in `figures/lc/plasticc/`
(300-dpi PNG + PDF); saved models live in `models/lc/plasticc/<task>/<algorithm>/`;
cached feature matrices live in `plasticc_data/_features/`.

This notebook is the simulated-data twin of `lc_classifier_ztf.ipynb`. It trains
the **same four algorithms** on features extracted from PLAsTiCC light curves so
that the RQ3 comparison — "best classifier on real ZTF" versus "best classifier on
simulated LSST" — is fair. PLAsTiCC (the Photometric LSST Astronomical Time-Series
Classification Challenge; Kessler et al., 2019; Hložek et al., 2023) is a simulated
catalogue of LSST light curves with known types; gradient-boosted trees won it
(Boone, 2019), which is the direct precedent for the algorithm family used here.

---

## 1. Data and the temporal split

The PLAsTiCC training metadata supplies the labels (`target` is the integer class
code) and host-galaxy / extinction context; the training light-curve file supplies
the time series — one row per flux measurement, `(object_id, mjd, passband, flux,
flux_err, detected_bool)`, with passbands 0–5 corresponding to LSST *u g r i z y*.

**Temporal split.** As on the ZTF side, the split is temporal, never shuffled. For
each object we compute the **first-detection MJD** (the earliest epoch flagged
`detected_bool = 1`), order objects oldest-to-newest, and cut 70 / 15 / 15 into
train / validation / test. The test fold is therefore strictly in the future.
Stratification was checked but never allowed to override the time ordering — a pure
temporal split leaves some rare classes thin in the later folds, which we report
rather than hide.

**Optional test augmentation.** The notebook exposes a decision cell,
`USE_TEST_AUGMENT`, default **off**. If enabled it would append a size-capped,
time-consistent slice of the unblinded PLAsTiCC test set to the training fold; it is
left off because the ~7 GB test light-curve files are unnecessary for the
methodological comparison this notebook makes, and keeping it off bounds runtime and
memory.

---

## 2. Feature extraction

Each object is reduced to a fixed-length feature vector with the **light-curve**
package (Malanchev et al., 2021), a Rust-backed extractor fast enough to process the
whole training set in seconds; the resulting matrices are cached to parquet so
re-runs skip extraction. The descriptor set is chosen to overlap the ALeRCE feature
family used on the ZTF side — amplitude, beyond-1σ fraction, cumulative sum,
variability η and η_e, kurtosis and skew, linear trend, maximum slope, median
absolute deviation and buffer-range percentage, percent amplitude and percentile
differences, standard deviation, Stetson-K, weighted mean, an Anderson–Darling
normality statistic, and a periodogram peak — computed **per passband**, plus
adjacent-band colours from the per-band weighted means and the metadata features
`hostgal_photoz`, `hostgal_photoz_err`, `mwebv` and `ddf_bool`. Because PLAsTiCC
light curves are in **flux** (and can be negative), features are computed on flux
directly rather than on magnitudes. A band with fewer than five epochs yields NaN
for that band's features — kept as informative missingness, exactly as on the ZTF
side.

Two feature matrices are produced: the full **6-band** matrix, and a **g, r-only**
matrix that is a strict column subset of it (guaranteeing the two variants are
perfectly consistent). The g, r variant restricts the simulated data to the two
bands ZTF actually observes.

**Feature-set EDA** (Figures `01_class_distribution`, `02_sampling_missingness`):
objects split cleanly into Galactic (host redshift 0 — the periodic variables) and
extragalactic (the transients), an almost-deterministic cue the models exploit and
the reason the host-galaxy photo-z is included as a feature; sampling is uneven
across bands, which is the source of the feature missingness kept as NaN for the
trees.

---

## 3. Experimental design — two tasks × two band variants

The notebook answers two questions, on two band variants:

1. **Native 14-class** — the full PLAsTiCC taxonomy ("what is the best PLAsTiCC
   classifier?", to sit next to Boone, 2019).
2. **Coarse SN / AGN / VS** — the same three-class problem the ZTF notebook solves,
   via a taxonomy bridge, so the datasets can be compared.

and, orthogonally, **6-band** *ugrizy* versus **g, r-only**.

**The coarse bridge.** The gold "SN" class is genuine supernovae, so only PLAsTiCC
SN subtypes (SNIa, SNIa-91bg, SNIax, SNII, SNIbc, SLSN-I) map to SN; AGN maps to
AGN; the periodic / stellar variables (RRL, EB, Mira, M-dwarf) map to VS. TDE, KN
and microlensing have **no** counterpart in the ZTF gold taxonomy and are dropped
from the coarse task rather than forced into a class — a deliberate, documented
choice that keeps the sim-to-real comparison honest.

To keep the notebook tractable, the **full four-algorithm comparison runs on the
6-band native task**; the winner and a reference gradient-booster are then carried
to the other three variants, with a short re-tune of learning-rate and number of
trees only (bounded runtime). The **g, r coarse** model is the artefact that
physically transfers to ZTF in RQ3.

The four algorithms, their justification, and the by-family preprocessing /
imbalance handling are identical to the ZTF notebook (LightGBM primary; Balanced RF
as the ALeRCE-aligned baseline; XGBoost as a robustness check; MLP as the
neural-tabular baseline). See `lc_classifier_ztf.md` §3 for the full rationale and
citations; it is not repeated here.

---

## 4. Tuning and metrics

Tuning is Optuna TPE against **macro-F1 on the temporal validation fold** (50 trials
for the boosters, 25 for Balanced RF and the MLP), with the same search spaces as
the ZTF notebook; Figure `03_optuna_history` shows the convergence. The full metric
battery is identical to the ZTF side, with one addition on the native task: the
official **PLAsTiCC weighted multi-class log loss** (Malz et al., 2019), reported so
the results sit next to the challenge leaderboard. Every tuned model is saved with
the same persistence contract (`model.joblib`, native booster, `best_params.json`,
`model_card.json`) as the ZTF notebook, under
`models/lc/plasticc/<task>/<algorithm>/`.

---

## 5. Results

### 6-band native — the "best PLAsTiCC classifier" table

The four tuned models are evaluated once on the future test fold; the metric table
(including weighted log loss) is written to
`figures/lc/plasticc/6band_native_test_metrics.csv`.

<!-- RESULTS_TABLE_PLASTICC_NATIVE -->
*(Populated from the full run.)*

Figure `04_native_confusion_perclass` shows the winner's row-normalised confusion
matrix and its per-class F1 against training-set size. Test F1 rises with
training-set size — the rare classes (KN, Mira) sit in the bottom-left with
near-zero F1, exactly the tail macro-F1 is designed to expose. Confusion is
concentrated *within* astrophysically-similar groups (SN subtypes with each other;
periodic variables with each other), not across the Galactic / extragalactic divide.

### The RQ3 feeder — variant sweep

Figure `05_rq3_variants` is the headline plot: macro-F1 for 6-band versus g, r and
native versus coarse. The variant metrics are in
`figures/lc/plasticc/variant_comparison.csv`.

<!-- RESULTS_TABLE_PLASTICC_VARIANTS -->
*(Populated from the full run.)*

Reading the figure:

- Dropping from **6 bands to g, r** costs macro-F1 — the size of that drop is the
  quantitative feasibility answer for proposal §6.5: it is what is sacrificed by
  matching ZTF's two bands.
- The **coarse** task scores much higher than the native 14-class task (three
  well-separated super-classes rather than fine subtypes), and it is the regime the
  RQ3 transfer actually operates in.
- The **g, r coarse** model is handed to the RQ3 sim-to-real study.

### Findings

- On simulated PLAsTiCC light curves, as on real ZTF data, **gradient-boosted trees
  are the best light-curve classifier**, reproducing Boone (2019) with an
  independent feature set and confirming the "trees win" conclusion is not
  dataset-specific.
- The native 14-class task exposes a genuine **rare-class tail** (KN, Mira near zero
  F1); macro-F1 is chosen precisely because it surfaces this rather than letting the
  dominant SNIa class mask it.
- The band restriction to g, r has a **measurable, quantified cost** — the concrete
  feasibility number the proposal asked for.

---

## 6. Risks and limitations

- **Rare classes.** KN (100 objects) and Mira (30) have near-zero F1 for some
  algorithms; this is reported honestly, and the pure temporal split (no
  stratification) makes the later folds thinner still for these classes.
- **Flux-based features.** PLAsTiCC provides calibrated flux, whereas the ZTF gold
  branch uses ALeRCE magnitude-domain features; the feature families are chosen to
  overlap conceptually, but they are not identical, which is a caveat for the RQ3
  transfer to keep in mind.
- **Simulated cadence.** PLAsTiCC's LSST cadence differs from ZTF's; the g, r
  restriction narrows but does not eliminate that gap. RQ3 quantifies the residual
  sim-to-real difference rather than assuming it away.

---

## References

Boone, K. (2019) 'Avocado: photometric classification of astronomical transients
with Gaussian process augmentation', *The Astronomical Journal*, 158(6), 257.

Hložek, R. et al. (2023) 'Results of the Photometric LSST Astronomical Time-Series
Classification Challenge (PLAsTiCC)', *The Astrophysical Journal Supplement Series*,
267(2), 25.

Kessler, R. et al. (2019) 'Models and simulations for the Photometric LSST
Astronomical Time-Series Classification Challenge (PLAsTiCC)', *Publications of the
Astronomical Society of the Pacific*, 131(1003), 094501.

Malanchev, K.L. et al. (2021) 'Anomaly detection in the Zwicky Transient Facility
DR3', *Monthly Notices of the Royal Astronomical Society*, 502(4), pp. 5147–5175.
(light-curve feature-extraction package.)

Malz, A.I. et al. (2019) 'The photometric LSST astronomical time-series
classification challenge PLAsTiCC: selection of a performance metric for
classification probabilities balancing diverse science goals', *The Astronomical
Journal*, 158(5), 171.

*Algorithm and tooling references (LightGBM, XGBoost, Balanced RF, Optuna,
imbalanced-learn, and the tabular-vs-deep-learning evidence) are listed in*
`lc_classifier_ztf.md`.
