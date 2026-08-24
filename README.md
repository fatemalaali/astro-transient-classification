# astro-transient-classification

Multimodal, real-time classification of ZTF transients into **SN / AGN / VS**, with a
late-fusion architecture over two independently trained branches, plus a
simulation-to-reality transfer study on PLAsTiCC and a live Fink-broker demo.

| Branch | Model | Input | Test macro-F1 |
|---|---|---|---:|
| Tabular | LightGBM | 242 ALeRCE light-curve features | 0.946 |
| Image | EfficientNet-B0 | 3 x 63 x 63 science/reference/difference cutouts | 0.759 |
| **Fusion** | multinomial logistic stack on branch log-probabilities | both | **0.953** |

**Label provenance is the methodological spine of the project.** Labels come only from
spectroscopic and catalogue sources (BTS, TNS, Chen+2020, Milliquas, SDSS DR16Q).
Brokers (ALeRCE, Fink) supply *alert packets, cutouts and precomputed features only* —
never classifications, and never anything derived from a broker's own model. The demo
enforces this at runtime (`demo/config.py: BROKER_URL_DENYLIST`, `tests/test_provenance.py`).

---

## 1. Quick orientation

Nothing in the repository uses absolute paths — every path resolves against the
repository root, so the checkout can live anywhere. Notebooks are run **from the
repository root**, not from a subdirectory.

Three trees are gitignored and must be produced or copied by hand: `data/`,
`plasticc_data/`, and `models/`. Sections 3–5 explain how to obtain each.

---

## 2. Environment

Python **3.10+** (the notebooks were run under 3.10.20 and 3.13.3). A GPU is strongly
recommended for `stamp_classifier_ztf.ipynb`; everything else is CPU-fine.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell, or
`source .venv/bin/activate` on macOS/Linux.

### 2.1 Credentials

```bash
cp .env.example .env
```

Then fill in the TNS bot credentials — see [.env.example](.env.example) for what each
variable does, and section 3.1 for how to obtain them. `.env` is gitignored.

### 2.2 Dependencies

Every notebook self-bootstraps its own dependencies in its first cell (a `pip install`
of only what is missing), so there is no research-side `requirements.txt`. To install
everything up front instead:

```bash
python -m pip install numpy pandas "pyarrow>=18" scikit-learn matplotlib seaborn tqdm requests python-dotenv alerce astroquery astropy lightgbm xgboost imbalanced-learn optuna light-curve torch torchvision timm
```

The **demo** has a pinned requirements file, because two of its pins are load-bearing
(`lightgbm==4.6.0` wrote `model.txt`; `fink-client==11.0` matches the `AlertConsumer`
internals the Kafka adapter touches):

```bash
python -m pip install -r requirements-demo.txt
```

On macOS, LightGBM needs OpenMP: `brew install libomp`.

---

## 3. Data access

| Source | What it provides | Auth | How it is fetched |
|---|---|---|---|
| [BTS Explorer](https://sites.astro.caltech.edu/ztf/bts/explorer.php) | ~20,500 spectroscopically classified ZTF transients, carrying `ZTFID` | none | `build_dataset.ipynb` → `data/bronze/bts.csv` |
| [TNS](https://www.wis-tns.org) public-objects dump | ~198,800 reported transients | **bot credentials** | `build_dataset.ipynb` → `data/bronze/tns_public_objects.csv` |
| [Chen+2020, `J/ApJS/249/18`](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/249/18) | ZTF periodic variable stars | none | VizieR via `astroquery` → `data/bronze/chen_vs.parquet` |
| [Milliquas v8, `VII/294`](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=VII/294) | Million Quasars catalogue (Flesch 2023) | none | VizieR via `astroquery` → `data/bronze/milliquas.parquet` |
| [SDSS DR16Q, `VII/289`](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=VII/289) | Quasar catalogue (Lyke+2020) | none | VizieR via `astroquery` → `data/bronze/sdss_dr16q.parquet` |
| [ALeRCE](https://alerce.online) ([API](https://api.alerce.online)) | ZTF `oid` resolution, 242 light-curve features, 63x63 stamp triplets, raw detections | none | `alerce` Python client, in `build_dataset.ipynb` and `fetch_ztf_lightcurves.ipynb` |
| [PLAsTiCC — Zenodo record 2539456](https://zenodo.org/records/2539456) | Simulated LSST light curves + metadata (~20 GB, unblinded) | none | `python download_plasticc.py` → `plasticc_data/` |
| [Fink broker](https://fink-broker.org) ([client docs](https://doc.ztf.fink-broker.org/services/fink_client/)) | Live ZTF alert stream over Kafka | **credentials by request** | demo only — section 6 |

### 3.1 TNS bot credentials (required for the dataset build)

1. Register a user account at <https://www.wis-tns.org> and confirm the e-mail.
2. In your profile, open **Bots** and add a bot. TNS issues a numeric **bot ID**; you
   choose the **bot name**; the bot's page shows the **API key**.
3. Put all three into `.env` as `TNS_BOT_ID`, `TNS_BOT_NAME`, `TNS_API_KEY`.

The dump is downloaded with a `tns_marker` User-Agent built from those values. Without
them the TNS stage is skipped with a warning and the dataset is built from BTS + VizieR
only — the pipeline still runs, but the class mix and object count differ from the
published ones, so **every number in `docs/` assumes TNS credentials were present**.

> `data/gold/MANIFEST.json` records the full `Config`, TNS API key included. `data/` is
> gitignored, but scrub that file before sharing the gold artefacts with anyone.

### 3.2 PLAsTiCC

```bash
python download_plasticc.py
```

Downloads the whole record (~20 GB) with resume support and MD5 verification;
re-running skips what is already present and valid. Also useful:

```bash
python download_plasticc.py --list
```

```bash
python download_plasticc.py --train-only
```

`--train-only` skips the eleven test light-curve chunks and is enough for the EDA; the
full download is needed for `lc_classifier_plasticc.ipynb`.

### 3.3 Fink Livestream (demo only)

1. Request credentials at <https://forms.gle/2td4jysT4e9pkf889>.
2. `pip install fink-client==11.0`
3. Register, substituting your own username and group id:

```bash
fink_client_register -survey ztf -username YOUR_USER -group_id YOUR_GROUP -mytopics fink_sn_candidates_ztf -servers kafka-ztf.fink-broker.org:24499 -maxtimeout 10 --verbose
```

4. Confirm `~/.finkclient/ztf_credentials.yml` exists and has `password: null`.

Fink credentials never go in `.env` — the client keeps them in that YAML file. To see
what is actually reachable from the current machine (Kafka, ALeRCE hosts, credential
files), run:

```bash
python scripts/check_connectivity.py
```

---

## 4. Order of execution

Each notebook is idempotent and cache-first: interrupted runs resume, and re-running a
completed stage reloads its artefacts instead of re-fetching. The dependencies below
are hard — a notebook asserts rather than silently running on a stale input.

```
              download_plasticc.py ─────────────────► plasticc_data/
                                                          │
1. build_dataset.ipynb ──► data/bronze ──► data/silver ──► data/gold
                                                │             │
2. eda.ipynb ◄──────────────────────────────────┘             │
3. eda_gold_plasticc.ipynb ◄──────────────────────────────────┤ (+ plasticc_data/)
                                                              │
4. lc_classifier_ztf.ipynb      ──► models/lc/ztf      ◄──────┤
5. lc_classifier_ztf_fine.ipynb ──► models/lc/ztf_fine ◄──────┤
6. stamp_classifier_ztf.ipynb   ──► models/stamp       ◄──────┤
              │                       │                       │
7. fusion_ztf.ipynb ◄─────────────────┘  ──► models/fusion    │
                                                              │
8. fetch_ztf_lightcurves.ipynb ──► data/rq3b_ztf_lc    ◄──────┘
              │
9. lc_classifier_plasticc.ipynb ──► models/lc/plasticc, models/rq3b
   (needs plasticc_data/, plus steps 4 and 8 for its RQ3b section)
```

| # | Notebook | Needs | Produces | Notes |
|---|---|---|---|---|
| 0 | `download_plasticc.py` | network | `plasticc_data/` | ~20 GB; any time before step 3 |
| 1 | `build_dataset.ipynb` | `.env` (TNS), network | `data/{bronze,silver,gold}` | The long one — thousands of ALeRCE fetches. Set `QUICK_TEST=1` for a ~50/class smoke run first. Resumable |
| 2 | `eda.ipynb` | `data/{bronze,silver}` | inline figures | Optional, read-only |
| 3 | `eda_gold_plasticc.ipynb` | `data/gold`, `plasticc_data/` | inline figures | Optional, read-only |
| 4 | `lc_classifier_ztf.ipynb` | `data/gold` | `models/lc/ztf/*`, `figures/lc/ztf/*` | Tabular branch. Emits the **branch contract** fusion consumes: `oof_proba.npy`, `oof_oids.npy`, `test_proba.npy`, `test_oids.npy`, `temperature.json`, `model_card.json` |
| 5 | `lc_classifier_ztf_fine.ipynb` | `data/gold` | `models/lc/ztf_fine/*`, `figures/lc/ztf_fine/*` | 9-class stretch objective + hierarchical variant. Not on the fusion path |
| 6 | `stamp_classifier_ztf.ipynb` | `data/gold` | `models/stamp/*`, `figures/stamp/*` | Image branch, **GPU**. Emits the same branch contract |
| 7 | `fusion_ztf.ipynb` | steps 4 + 6 | `models/fusion/logreg_stack/*`, `figures/fusion/*` | Reads only the saved probability arrays — no LightGBM/torch training stack needed |
| 8 | `fetch_ztf_lightcurves.ipynb` | `data/gold` | `data/rq3b_ztf_lc/*` | Raw per-epoch ZTF photometry for all 11,826 gold objects. ~93 min at the polite 90 req/min pace; cached per object. `RQ3B_QUICK=1` for 60 objects |
| 9 | `lc_classifier_plasticc.ipynb` | `plasticc_data/`, + steps 4 & 8 for §9 | `models/lc/plasticc/*`, `models/rq3b/*`, `figures/lc/plasticc/*`, `figures/rq3b/*` | §1–8 need PLAsTiCC only. §9 (RQ3b sim-to-real transfer) also reads `data/rq3b_ztf_lc/` and asserts the `models/lc/ztf` model cards carry the same `split_id` |

Steps 4, 5 and 6 are mutually independent and can be run in any order, or in parallel.

### 4.1 The shared protocol

[protocol.py](protocol.py) is the single source of truth every branch notebook obeys, in
this order: **fit** on train → **tune** on validation → **select** the winner on
validation macro-F1 → **forward-chaining OOF** inside train → **fit temperature and the
fusion meta-learner on that OOF** → **refit** on train+val → **read test exactly once**.
It also defines `SEED = 42`, `COARSE_CLASSES`, and `split_id()` — the canonical hash of
the `oid → split` map that lets every notebook assert it trained on the same partition
(currently `76c4c40d0352`). Do not bypass it.

### 4.2 Fast paths

| Variable | Effect |
|---|---|
| `QUICK_TEST=1` | `build_dataset.ipynb`: ~50 objects per class |
| `LC_QUICK=1` | classifier + fusion notebooks: fewer Optuna trials, 200 bootstrap resamples |
| `RQ3B_QUICK=1` | `fetch_ztf_lightcurves.ipynb`: 60 objects |

---

## 5. Required file structure

Everything below is tracked in git except the three trees marked `(gitignored)` — those
you generate (sections 3–4) or copy between machines. Directories are created on demand;
nothing under `data/`, `models/` or `figures/` needs to be pre-created.

```
astro-transient-classification/
├── .env                              (gitignored)  ← your filled-in copy of .env.example
├── .env.example
├── protocol.py                       shared train/tune/select/calibrate protocol
├── download_plasticc.py
├── requirements-demo.txt
│
├── build_dataset.ipynb               ┐
├── eda.ipynb                         │
├── eda_gold_plasticc.ipynb           │
├── lc_classifier_ztf.ipynb           │  the experiment, in run order —
├── lc_classifier_ztf_fine.ipynb      │  see section 4
├── stamp_classifier_ztf.ipynb        │
├── fusion_ztf.ipynb                  │
├── fetch_ztf_lightcurves.ipynb       │
├── lc_classifier_plasticc.ipynb      ┘
│
├── data/                             (gitignored — built by the notebooks)
│   ├── bronze/                       raw catalogue downloads
│   │   ├── bts.csv
│   │   ├── tns_public_objects.csv
│   │   ├── chen_vs.parquet
│   │   ├── milliquas.parquet
│   │   └── sdss_dr16q.parquet
│   ├── silver/
│   │   ├── labelled_objects.parquet  one cleaned, deduplicated, oid-resolved table
│   │   └── _oid_cache.json           ALeRCE cone-search cache
│   ├── gold/                         training-ready artefacts — 11,826 objects
│   │   ├── gold_features.parquet     267 columns
│   │   ├── gold_stamps.npz           N x 3 x 63 x 63
│   │   ├── gold_labels.parquet       coarse + fine
│   │   ├── gold_metadata.parquet     ndet, redshift, sky position
│   │   ├── gold_splits.parquet       time-ordered train 0.70 / val 0.15 / test 0.15
│   │   ├── MANIFEST.json             config, counts, split_id
│   │   ├── _stamp_norm.npy           cached normalised stamp tensor
│   │   └── _cache_features/  _cache_meta/  _cache_stamps/    per-object, resumable
│   ├── rq3b_ztf_lc/                  raw ZTF photometry for the RQ3b transfer study
│   │   ├── rq3b_ztf_detections.parquet              1.3M detections
│   │   ├── rq3b_ztf_detections_manifest.json
│   │   ├── rq3b_features_{ztf,plasticc}_{raw,norm}.parquet
│   │   └── _cache_detections/
│   ├── demo/                         demo runtime state — demo.db, stamps/, raw_alerts/
│   └── pipeline.log
│
├── plasticc_data/                    (gitignored — `python download_plasticc.py`)
│   ├── plasticc_train_lightcurves.csv.gz
│   ├── plasticc_train_metadata.csv.gz
│   ├── plasticc_test_lightcurves_01..11.csv.gz
│   ├── plasticc_test_metadata.csv.gz
│   ├── note1_dataRelease.pdf   note2_modelNames.pdf   plasticc_modelpar.tar.gz
│   └── _features/                    extracted feature cache (train_6band.parquet)
│
├── models/                           (gitignored — ~100 MB, produced by steps 4–9)
│   ├── lc/ztf/{lightgbm,xgboost,brf,mlp}/     model.joblib, model_card.json,
│   │                                          best_params.json — and, on the winner,
│   │                                          oof_proba.npy, oof_oids.npy,
│   │                                          val_proba.npy, test_proba.npy,
│   │                                          test_oids.npy, temperature.json
│   ├── lc/ztf_fine/{lightgbm,xgboost,brf,hierarchical/}
│   ├── lc/plasticc/{6band_native,6band_coarse,gr_native,gr_coarse}/{lightgbm,xgboost,...}
│   ├── stamp/{effnet_b0,resnet18,rotcnn}/     state_dict.pt, model_scripted.pt,
│   │                                          preprocess.json, model_card.json
│   │                                          (+ the branch contract, on the winner)
│   ├── fusion/logreg_stack/          meta_learner.pt, fusion_card.json, best_params.json
│   └── rq3b/rq3b_transfer_card.json
│
├── figures/                          tracked — publication figures (PNG + PDF) and the
│   ├── lc/{ztf,ztf_fine,plasticc}/   CSVs behind every reported number
│   ├── stamp/   fusion/   rq3b/
│
├── docs/
│   ├── dataset_methodology.md        how the gold layer is built, and why
│   ├── lc_classifier_ztf.md   lc_classifier_ztf_fine.md   lc_classifier_plasticc.md
│   ├── stamp_classifier_ztf.md   fusion_ztf.md
│   └── demo-guide.md                 running and understanding the live demo
│
├── demo/                             live Fink demo — section 6
│   ├── adapters/                     fink_kafka, fink_rest, replay, cutouts, offsets
│   ├── ingest/   inference/   api/   storage/   web/
│   └── config.py   models.py   run_api.py   run_consumer.py
├── config/
│   ├── demo.env.example              annotated demo settings
│   ├── demo.env                      (gitignored) your copy
│   └── replay_manifest.json          pinned Kafka offsets — committed on purpose, so
│                                     the replay demo is reproducible from a checkout
├── scripts/                          seed_demo_db, check_connectivity, backfill_features,
│                                     record_replay_manifest, verify_cutouts,
│                                     compare_stamp_orientation
└── tests/                            pytest — adapters, provenance, serving contract,
                                      backfill
```

---

## 6. The live demo (optional)

Full instructions in [docs/demo-guide.md](docs/demo-guide.md). The safe path needs no
credentials and no network — it replays real held-out objects out of the gold layer:

```bash
python scripts/seed_demo_db.py --n 250
```

```bash
python -m demo.run_api --open
```

The live path is two processes in two terminals — the consumer plus the API above:

```bash
python -m demo.run_consumer --mode live
```

Ingestion modes: `live` (Kafka, seeks to the end of each partition), `catchup` (Kafka
backlog, throttled), `replay` (Kafka at the pinned offsets in
`config/replay_manifest.json` — identical every run), `rest` (Fink REST polling, no
credentials), `offline` (archived `.avro` files, no network). `DEMO_USE_STUBS=1` runs
the whole stack with no model artefacts at all. Settings live in `config/demo.env`
(copy `config/demo.env.example`) or in `.env`.

To move the demo to another machine, copy the repository plus `models/` (~100 MB) and
`data/gold/` (for seeding), then re-run `fink_client_register`.

```bash
python -m pytest tests -q
```

---

## 7. Reproducing a specific reported number

Every figure has its numbers written beside it as CSV. `figures/fusion/test_metrics.csv`
and `figures/fusion/verdict.csv` carry the headline fusion result;
`figures/lc/ztf/test_metrics.csv` and `figures/stamp/test_metrics.csv` the two branches;
`figures/rq3b/rq3b_transfer_metrics.csv` the sim-to-real study. The matching narrative
for each notebook lives in `docs/<notebook>.md`.
