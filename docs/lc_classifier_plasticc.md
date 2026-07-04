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

From the full run (50/50/25/25 Optuna trials), on the future test fold:

| Model | Macro-F1 | Balanced acc. | Accuracy | MCC | ROC-AUC | PR-AUC | Log loss | Weighted log loss |
|---|---|---|---|---|---|---|---|---|
| **XGBoost** | **0.549** | 0.667 | 0.668 | 0.553 | 0.933 | 0.725 | 0.972 | **0.960** |
| LightGBM | 0.537 | 0.640 | 0.670 | 0.545 | 0.930 | 0.728 | 0.917 | 1.038 |
| Balanced RF | 0.456 | 0.650 | 0.498 | 0.398 | 0.898 | 0.623 | 1.474 | 1.145 |
| MLP | 0.414 | 0.534 | 0.562 | 0.407 | 0.885 | 0.544 | 2.958 | 3.779 |

**XGBoost wins the 6-band native task** (macro-F1 0.549, weighted log loss 0.960),
narrowly ahead of LightGBM (0.537) — the two gradient-boosted trees again lead
clearly over Balanced RF (0.456) and the MLP (0.414). The absolute macro-F1 is
modest because this is the full 14-class problem with a heavy rare-class tail;
XGBoost is carried, with LightGBM as the reference, to the variant sweep.

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

Test macro-F1 for the two carried models across the four variants:

| Task | Bands | Labels | XGBoost | LightGBM |
|---|---|---|---|---|
| 6-band native | *ugrizy* | 14-class | 0.549 | 0.537 |
| g, r native | g, r | 14-class | 0.467 | 0.468 |
| 6-band coarse | *ugrizy* | SN/AGN/VS | 1.000 | 1.000 |
| **g, r coarse** | **g, r** | **SN/AGN/VS** | **0.889** | **0.889** |

Reading the figure:

- Dropping from **6 bands to g, r** costs macro-F1: ≈ 0.08 on the native task
  (0.55 → 0.47) and ≈ 0.11 on the coarse task (1.00 → 0.89). That drop is the
  quantitative feasibility answer for proposal §6.5 — what is sacrificed by matching
  ZTF's two bands.
- The **coarse** task scores far higher than the native 14-class task, and it is the
  regime the RQ3 transfer actually operates in.
- The **g, r coarse** model (macro-F1 0.889) is the artefact handed to the RQ3
  sim-to-real study.

**Important caveat on the coarse scores.** The pure temporal split interacts with
astrophysics in a way that must be stated. AGN are *persistent* variables detected
almost as soon as the survey begins, so their first-detection epochs cluster early
and they land overwhelmingly in the training fold; supernovae are transient and
spread across time, so they dominate the later folds. The coarse **test** fold is
therefore 992 SN, 38 VS and only **2 AGN**. Combined with the fact that the VS class
has `hostgal_photoz` identically zero (deterministically Galactic), the 6-band
coarse problem becomes trivially separable on that particular test fold — which is
why it reaches a *perfect* 1.000. That figure should be read as "simulated coarse
classes are cleanly separable given all six bands", not as a robust generalisation
estimate; the **g, r coarse** number (0.889), where the band restriction breaks the
clean separation, is the more informative and honest result, and it is the one RQ3
uses. We flag the AGN scarcity in the later folds as a limitation of the temporal
split (§6).

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
- **AGN concentration under the temporal split.** Because AGN are persistent and
  detected early, the temporal ordering pushes almost all of them into the training
  fold, leaving only 2 AGN in the coarse test fold. The coarse-task scores — the
  perfect 6-band figure especially — therefore rest on very few minority objects and
  should be read with the g, r coarse result (0.889), not the 1.000, as the
  representative number (see §5).
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
