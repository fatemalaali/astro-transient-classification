# Dataset Construction Methodology

*A medallion (bronze → silver → gold) pipeline for a training-ready, multimodal
dataset for real-time transient classification on ZTF.*

This document explains what the `build_dataset.ipynb` pipeline does at each
layer, both **technically** (the engineering: what code runs, what files are
produced) and **academically** (why each choice is defensible and how it serves
the research). It is meant to be read alongside the notebook, not as a
replacement for it.

---

## 0. Why this pipeline exists

The research goal is to **classify ZTF transients in real time from multimodal
inputs** — combining tabular light-curve features with image cutouts — and to
**benchmark** that against existing brokers (e.g. the ALeRCE stamp classifier).

That benchmarking goal imposes one hard constraint that shapes the entire
design:

> **Labels must come only from sources that are independent of any broker's
> classifier.**

If the training labels were taken from a broker's own predictions, any
subsequent comparison against that broker would be circular — the model would
be graded against the very system that taught it. To avoid this, the pipeline
draws ground truth exclusively from **spectroscopic surveys and independent
astronomical catalogues**, and uses the broker (ALeRCE) **only as a data
delivery service** to fetch features, light curves, and image stamps for
objects whose labels were *already fixed elsewhere*.

The **medallion architecture** (bronze → silver → gold) is a standard
data-engineering pattern that separates concerns cleanly:

| Layer | Mandate | Mutates labels? | Resumable unit |
|---|---|---|---|
| **Bronze** | Faithful capture of raw sources | No | Per-source file cache |
| **Silver** | Clean, unify, label, resolve IDs, dedupe | Yes (assigns labels) | Per-coordinate oid cache |
| **Gold** | Attach ML inputs, split, package | No | Per-object feature/stamp cache |

The value of the separation is **reproducibility and auditability**: each layer
is independently inspectable, the provenance of every label is preserved in a
`source` column, and every stage is cached so a multi-hour run survives
interruption.

---

## 1. Bronze — faithful capture of raw sources

### What happens technically

The bronze layer downloads each raw catalogue exactly as published and caches
it to `data/bronze/`. There is **no cleaning, joining, or relabelling** — the
mandate is fidelity to source. Every loader is cache-first: if the file already
exists locally, it is read back; otherwise it is downloaded and saved.

Five sources are pulled:

| Source | Access method | File | Rows (full run) |
|---|---|---|---|
| **BTS** — ZTF Bright Transient Survey | HTTP CSV export | `bts.csv` | ~20,500 |
| **TNS** — Transient Name Server public objects | Bot-authenticated POST (zip) | `tns_public_objects.csv` | ~198,800 |
| **Chen+2020** — ZTF Catalog of Periodic Variable Stars (`J/ApJS/249/18`) | VizieR via `astroquery` | `chen_vs.parquet` | ~200,000 |
| **Milliquas v8** — Million Quasars, Flesch 2023 (`VII/294`) | VizieR via `astroquery` | `milliquas.parquet` | ~150,000 |
| **SDSS DR16Q** — quasar catalogue, Lyke+2020 (`VII/289`) | VizieR via `astroquery` | `sdss_dr16q.parquet` | ~150,000 |

Implementation notes worth knowing:
- **TNS** requires bot credentials (`TNS_BOT_ID/NAME/API_KEY`, loaded from `.env`)
  and a specially-formatted `User-Agent` marker; its dump has a metadata comment
  on line 1, so the real header is on line 2 (`skiprows=1`).
- The three VizieR catalogues are pulled with a generous `row_limit` (e.g.
  200k) for the full run, or a small cap in `QUICK_TEST` mode.
- Output formats differ by convenience: CSV for the survey dumps, Parquet for
  the large VizieR tables.

### What it means academically

Each source is chosen because it is an **independent label authority** for one
of the target classes, and because its labels are *trustworthy by
construction*:

- **BTS and TNS** provide **spectroscopic ground truth** for transients —
  classifications confirmed by a spectrum, the gold standard in transient
  astronomy. BTS is magnitude-limited and highly complete for bright SNe;
  TNS is the central registry of reported transients and contributes fainter
  objects and rarer types (TDEs, etc.).
- **Chen+2020** is a large, vetted catalogue of **periodic variable stars**
  derived from ZTF itself, giving high-purity VS labels.
- **Milliquas and SDSS DR16Q** are well-established **quasar/AGN** catalogues;
  AGN labels here rest on spectroscopy and multi-wavelength evidence, not on a
  transient broker's guess.

By keeping the bronze layer a verbatim mirror, the pipeline guarantees that the
**raw provenance of every object is preserved and re-derivable**, which is a
prerequisite for a defensible methods section: a reviewer can point at any
training example and trace it back to a named, citable catalogue.

---

## 2. Silver — clean, unify, label, resolve IDs, deduplicate

### What happens technically

The silver layer collapses five heterogeneous catalogues into **one unified,
labelled object table** with a fixed schema:

```
source, ext_name, ra, dec, redshift, raw_type, coarse, fine, oid
```

Five operations run in sequence:

**(1) Standardize.** Each source has a `_std_*` function that maps its
idiosyncratic columns onto the unified schema. A helper, `_find_col`, tolerates
the fact that RA/Dec/type columns are named differently in every catalogue
(`RAJ2000`, `RA_ICRS`, `radeg`, …). Coordinates are normalized to decimal
degrees via `_to_deg_ra`/`_to_deg_dec`, which accept either degrees or
sexagesimal (`hh:mm:ss`) strings.

**(2) Label with a two-level taxonomy.** `map_label` converts each raw
classification string into:
- **`coarse`** — the head-to-head training classes: **SN / AGN / VS**
  (supernova, active galactic nucleus, variable star). These are the classes
  the model is trained and ablated on, and the ones comparable to the ALeRCE
  stamp classifier.
- **`fine`** — detailed subtype (e.g. `SN Ia-91T`, `SLSN`, `Blazar`, `RRL`,
  `EB`, `Mira`), used for the stretch objective.
- A separate `_PLASTICC_MAP` adds **`plasticc_class`**, aligning fine labels
  with PLAsTiCC's simulated-data class names for the simulated-vs-real
  comparison (RQ3). Only unambiguous correspondences are mapped.

Mapping uses curated keyword and regex rules (e.g. SN subtype patterns,
AGN-family keywords, a variable-star code dictionary). Objects that don't map to
one of the coarse classes get `coarse=None`.

**(3) Sample to per-class caps.** `build_silver` draws up to configured caps —
`sn_max=8000`, `agn_max=5000`, `vs_max=5000`, plus `tns_extra_max=4000` — so no
single abundant class dominates. AGN are pooled from Milliquas + SDSS before
sampling.

**(4) Resolve the ZTF `oid`** (the slow, critical step, `resolve_oids`). Every
object needs a ZTF object ID to fetch data downstream:
- BTS rows already carry the `ZTFID`.
- TNS rows may have one embedded in `internal_names` (extracted by the
  `ZTF\d{2}[a-z]{7}` regex).
- For the variable-star and AGN catalogues (which have no ZTF id), the pipeline
  runs an **ALeRCE cone search** on `(ra, dec)` within `xmatch_radius_arcsec =
  1.5″` and takes the matched object with the most detections.

Objects with no ZTF match are **dropped** — they cannot be used. Resolutions are
cached to `data/silver/_oid_cache.json` keyed by rounded coordinates, with
periodic checkpointing (every 500 lookups) so an interrupted run resumes. Misses
are deliberately *not* cached, so a transient lookup failure is retried next run
rather than permanently discarding the object.

**(5) Deduplicate by `oid`** using a **source-priority order**
(`bts > tns > chen > sdss > milliquas`): when the same physical object appears
in several catalogues, the highest-priority (most trustworthy) label is kept.

Output: `data/silver/labelled_objects.parquet` — one row per unique ZTF object,
with a clean coarse/fine label, redshift where available, and a resolved oid.

### What it means academically

This is where **measurement becomes data**. Several choices are methodologically
load-bearing:

- **The two-level taxonomy** mirrors how the field actually evaluates
  classifiers: a coarse SN/AGN/VS scheme that is directly comparable to existing
  broker outputs and robust to label noise, plus a fine scheme that supports a
  more ambitious sub-typing objective without contaminating the headline
  results. Keeping exotic transients (TDE, KN, novae) in `fine` but out of
  `coarse` (`include_exotic_in_coarse=False`) avoids polluting the main classes
  with rare, ambiguous objects while preserving them for later analysis.
- **Cone-search cross-matching at 1.5″** is the standard way to associate a
  catalogue position with a ZTF source; the tight radius trades a little
  completeness for **purity** (fewer chance mismatches), which is the right
  trade-off when label quality matters more than sample size.
- **Source-priority deduplication** encodes an explicit, documented trust
  hierarchy — spectroscopic transient surveys over registry entries over
  catalogue cross-matches — so conflicting labels are resolved transparently
  rather than arbitrarily.
- **Per-class caps and sampling** address **class imbalance** at the source,
  preventing the naturally over-represented classes from biasing training and
  keeping compute tractable.

The net result is a single, citable, reproducible ground-truth table whose every
label can be defended as broker-independent.

---

## 3. Gold — attach ML inputs, split, and package

### What happens technically

The gold layer takes the silver object list and fetches the **actual model
inputs** from ALeRCE for each `oid`, in parallel
(`ThreadPoolExecutor`, `workers=6`), with per-object caching so the stage is
fully resumable. For every object it pulls three things:

- **Metadata** (`_fetch_meta`): `ndet`, `firstmjd`, `lastmjd`. Objects with
  fewer than `min_detections = 5` are rejected — too little light-curve signal.
- **Tabular features** (`_fetch_features`): ALeRCE's light-curve features,
  pivoted from long to wide as `<name>_<fid>` columns (fid = photometric band),
  so each object becomes one feature row.
- **Image stamps** (`_fetch_stamp`): the three ZTF cutout planes —
  **science, reference (template), and difference** — each NaN-cleaned and
  centre-cropped/zero-padded to `cutout_size = 63` px, stacked into a
  `(3, 63, 63)` tensor.

A **multimodal completeness filter** then keeps only objects that have **both** a
feature vector **and** a stamp. This is what makes the dataset genuinely
multimodal rather than a union of two partially-overlapping sets.

The surviving objects are assembled and written to `data/gold/`:

| Artefact | Contents |
|---|---|
| `gold_features.parquet` | `N × F` tabular feature matrix, keyed by oid |
| `gold_stamps.npz` | image tensor `(N, 3, 63, 63)` + oids + channel names |
| `gold_labels.parquet` | oid → `coarse`, `fine`, `plasticc_class` |
| `gold_metadata.parquet` | oid → ra, dec, redshift, source, ndet, firstmjd, lastmjd |
| `gold_splits.parquet` | oid → `train` / `val` / `test`, `firstmjd`, and the canonical `split_id` |
| `MANIFEST.json` | config snapshot + all counts (shapes, class/split balance, sources) + `split_id` |

The **`split_id`** is a canonical, order-independent hash of the sorted
`oid → split` assignment (identical to `protocol.split_id`), written into both
`gold_splits.parquet` and `MANIFEST.json`. It is the single source of split
identity: every downstream branch recomputes it and asserts it trained on this
exact partition, and the fusion notebook uses it (via `protocol.assert_same_split`)
to confirm all branches are fusable. It replaces the earlier per-notebook
`pd.util.hash_pandas_object` hashes, which differed cosmetically (with `index=` and
row order) and so could not be compared directly.

### The resulting classes (label taxonomy)

Every gold object carries **three label fields** (`gold_labels.parquet`),
produced by `map_label` / `_PLASTICC_MAP` in silver and copied through to gold:

- **`coarse`** — the head-to-head training classes used for the main ablation
  and the broker comparison.
- **`fine`** — the detailed subtype, used for the stretch (sub-typing) objective.
- **`plasticc_class`** — an optional mapping onto PLAsTiCC's simulated-data class
  names, used only for the simulated-vs-real comparison (RQ3).

**Coarse taxonomy** — exactly three classes (`cfg.coarse_classes = ("SN", "AGN",
"VS")`); everything else is excluded from coarse training:

| `coarse` | Meaning | Primary label sources |
|---|---|---|
| **SN** | Supernova (extragalactic transient) | BTS, TNS (spectroscopic) |
| **AGN** | Active galactic nucleus / quasar | Milliquas, SDSS DR16Q, TNS |
| **VS** | Variable star (Galactic, periodic/stochastic) | Chen+2020, TNS |

**Fine taxonomy** — the subtype each object resolves to, grouped by its coarse
parent:

| Coarse | `fine` classes |
|---|---|
| **SN** | `SN Ia`, `SN Ia-91T`, `SN Ia-91bg`, `SN Ia-CSM`, `SN Iax`, `SN Ib`, `SN Ic`, `SN Ibc`, `SN Ic-BL`, `SN II`, `SN IIP`, `SN IIn`, `SN IIb`, `SLSN`, `SN` (generic fallback) |
| **AGN** | `AGN/QSO`, `Blazar` |
| **VS** | `RRL` (RR Lyrae), `EB` (eclipsing binary), `DSCT` (δ Scuti), `Cepheid`, `Mira`, `LPV` (long-period variable), `CV` (cataclysmic variable), `YSO` (young stellar object), `VarOther` |
| **Exotic** *(kept in `fine`, `coarse = None` by default)* | `TDE`, `KN` (kilonova), `Nova/Impostor` |

> The exotic transients are retained in the table for later analysis but are
> **dropped from coarse training** because `cfg.include_exotic_in_coarse =
> False`. Flip that flag to fold TDE/KN/Nova into the SN coarse class. Anything
> that matches no rule lands in `fine = other:<...>` and `coarse = None`, so it
> never enters the coarse training set.

**PLAsTiCC mapping** (`plasticc_class`) — only the unambiguous correspondences
are mapped; everything else is left blank:

| `fine` → | `plasticc_class` |
|---|---|
| `SN Ia` | `SNIa` |
| `SN Iax` | `SNIax` |
| `SN Ia-91bg` | `SNIa-91bg` |
| `SN II`, `SN IIP`, `SN IIn`, `SN IIb` | `SNII` |
| `SN Ibc`, `SN Ib`, `SN Ic`, `SN Ic-BL` | `SNIbc` |
| `SLSN` | `SLSN-I` |
| `TDE` | `TDE` |
| `KN` | `KN` |
| `AGN/QSO`, `Blazar` | `AGN` |
| `RRL` | `RRL` |
| `EB` | `EB` |
| `Mira` | `Mira` |

### The features (gold model inputs)

Each gold object is described by **three modalities**, kept in separate
artefacts but aligned one-to-one by `oid`:

| Modality | Artefact | Shape (per object) | Source |
|---|---|---|---|
| **Tabular light-curve features** | `gold_features.parquet` | `F` scalar columns | ALeRCE `query_features` |
| **Image stamps** | `gold_stamps.npz` | `(3, 63, 63)` | ALeRCE `get_stamps` (ZTF cutouts) |
| **Metadata / context** | `gold_metadata.parquet` | 8 scalar fields | ALeRCE `query_object` + silver |

**Image channels** — the three standard ZTF cutout planes, stacked in this
fixed order:

| Index | Channel | What it is |
|---|---|---|
| 0 | `science` | the new science exposure cutout |
| 1 | `reference` | the deep template (reference) image |
| 2 | `difference` | science − reference (where transient flux appears) |

**Metadata fields** (`gold_metadata.parquet`):

| Field | Meaning / use |
|---|---|
| `ra`, `dec` | sky position (degrees) |
| `redshift` | redshift where the source catalogue provided one |
| `source` | which catalogue supplied the label (provenance) |
| `ndet` | number of detections (quality cut: `≥ 5`) |
| `firstmjd` | first detection time — **drives the time-ordered split** |
| `lastmjd` | last detection time |

**Tabular feature set.** The columns in `gold_features.parquet` are exactly
whatever ALeRCE's `query_features` returns for each object — i.e. the **ALeRCE
ZTF light-curve feature set** (the same features that power their Light Curve
Classifier). Each feature is pivoted to a wide column named `<name>_<fid>`,
where `fid` is the photometric band (`1` = ZTF *g*, `2` = ZTF *r*); band-agnostic
features keep a single column. The exact count `F` is determined at fetch time
and recorded as `n_features` in `MANIFEST.json`. The features fall into these
families:

| Family | Representative features | What it captures |
|---|---|---|
| **Detection statistics** | `n_det`, `n_pos`, `n_neg`, positive/negative fraction | how much and what kind of photometry exists |
| **Magnitude statistics** | `Mean`, `Std`, `Amplitude`, `IQR`, `Skew`, `(Small)Kurtosis`, `Beyond1Std`, `MaxSlope` | shape and spread of the brightness distribution |
| **Stochastic variability** | `StetsonK`, `Pvar`, `ExcessVar`, `Eta_e`, `AndersonDarling` | is the source genuinely varying, and how |
| **MHPS** (Mexican-Hat Power Spectrum) | `MHPS_ratio`, `MHPS_low`, `MHPS_high` | variability power on short vs long timescales |
| **Periodic features** | `Multiband_period`, `Period_band`, `Power_rate`, `Psi_CS`, `Psi_eta`, harmonic amplitudes/phases | periodicity — key for variable stars |
| **GP-DRW** (damped random walk) | `GP_DRW_tau`, `GP_DRW_sigma` | stochastic timescale/amplitude — key for AGN |
| **SPM** (Supernova Parametric Model fit) | `SPM_A`, `SPM_t0`, `SPM_gamma`, `SPM_beta`, `SPM_tau_rise`, `SPM_tau_fall`, `SPM_chi` | rise/fade light-curve shape — key for SNe |
| **Color** | `g-r` mean / max, color variation | spectral-energy-distribution proxy across bands |
| **Context / position** | `gal_b`, `gal_l`, WISE colors, star–galaxy score (`sgscore`) | Galactic vs extragalactic prior, host context |

> The family table is representative of the ALeRCE feature set, not a hard-coded
> schema: the pipeline stores whatever keys come back, so the precise column list
> can shift with the ALeRCE version. Always treat `MANIFEST.json → n_features`
> and the actual `gold_features.parquet` header as ground truth.

**The train/val/test split** (`make_splits`) is **time-ordered, not random**:
objects are sorted by `firstmjd` (first detection time); the oldest
`train_frac = 70%` go to train, the next `val_frac = 15%` to val, and the newest
15% to test. Objects with no `firstmjd` are treated as most-recent.

A final, deliberately disabled hook (`add_bogus_from_rb`) documents how a
"bogus" class *could* be built from the ZTF real-bogus score — and why it is
intentionally left out (see §4).

### What it means academically

The gold layer is where the dataset is shaped to **support honest evaluation**:

- **Multimodality with a strict completeness filter** is what the central
  research question requires: the model must learn from features *and* images
  jointly, so an object missing either modality would silently weaken the
  premise. Enforcing the intersection keeps the comparison between
  unimodal and multimodal models fair.
- **The minimum-detection cut** ensures every light-curve feature vector is
  computed from enough photometry to be meaningful, reducing label-independent
  noise.
- **The time-ordered split is the single most important methodological choice
  in this layer.** Two properties follow from it:
  1. **No object leakage** — each object lives wholly in one split, so the model
     can never be tested on an object it partly saw in training.
  2. **Temporal realism** — because the test set is strictly "in the future"
     relative to training, the evaluation **mimics deployment**, where a
     real-time classifier must label objects it has never seen, discovered after
     the model was trained. A random split would leak future information and
     inflate reported performance — a well-known pitfall in time-domain
     astronomy. The time-ordered split produces a **conservative, deployment-
     faithful estimate** of real-world accuracy.
- **The MANIFEST** records the exact configuration and resulting class/split
  balance, making any published result reproducible and the dataset's
  composition transparent to reviewers.

---

## 4. Two deliberate omissions (and why they matter)

- **No "bogus" class.** A clean bogus (junk-detection) label can only come from
  the ZTF real-bogus score, which is a **pipeline data-quality flag, not an
  independent science classification**. Including it would violate the
  broker-independence rule that underpins the whole design. The
  `add_bogus_from_rb` function is therefore an explicit, documented stub that
  raises `NotImplementedError`, and the proposal drops bogus from RQ1
  accordingly. This is an *honesty* choice: better to omit a class than to
  smuggle in a non-independent label.
- **`QUICK_TEST` mode** shrinks every cap to ~50/class, allowing a fast
  end-to-end smoke test of the full pipeline before committing to the
  multi-hour production pull. This supports iterative development without
  changing the production logic.

---

## 5. End-to-end summary

> **Bronze** caches five independent raw catalogues verbatim. **Silver** fuses
> them into one deduplicated, broker-independent, ground-truth-labelled object
> table with resolved ZTF ids. **Gold** attaches the multimodal light-curve
> features and image stamps, filters to objects complete in both modalities, and
> packages everything — with a leakage-free, time-ordered train/val/test split —
> into training-ready files plus a reproducibility manifest.

Every layer is cached and resumable; every label is traceable to a named,
broker-independent source; and the final split is built to estimate real-world,
real-time deployment performance rather than an optimistic in-sample number.
