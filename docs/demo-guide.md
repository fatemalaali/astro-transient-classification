# Demo Guide — what it does, how to run it, how the code works

Companion to `docs/demo-plan.md` (the design and the verified findings). This
document is the practical one: run it, understand the dashboard, understand the
code, move it to another machine.

---

## 1. What this demo is

It classifies **real ZTF alerts, live**, into **SN / AGN / VS**, by combining two
independently trained models:

| Branch | Model | Input | Held-out macro-F1 |
|---|---|---|---:|
| Tabular | LightGBM | 242 light-curve features | 0.946 |
| Image | EfficientNet-B0 | 3 × 63 × 63 science/reference/difference cutouts | 0.759 |
| **Fusion** | multinomial logistic stack on log-probabilities | both branches | **0.953** |

Alerts arrive from the **Fink broker** over Apache Kafka. Each one is decoded,
its cutout triplet extracted, its light-curve features resolved, both models
run, and the results fused — then written to a database that the dashboard reads.

**The methodological point the demo exists to make:** the class labels come from
spectroscopic and catalogue sources (TNS, BTS, Chen+2020, Milliquas, SDSS DR16Q).
Brokers supply *alert packets, cutouts and features only* — never labels, and
never model input derived from their own classifications. The interface shows
broker values in amber and marks them "not model input"; the code enforces it
(`tests/test_provenance.py`).

---

## 2. Quick start

### The safest start (no credentials, no network)

```bash
python scripts/seed_demo_db.py --n 250
python -m demo.run_api --open
```

That fills the database with 250 real objects from the held-out test fold — real
stamps, real features, real labels — and opens the dashboard. Good for
developing, and it is the demo-day fallback.

### The live demo

Two processes, two terminals:

```bash
python -m demo.run_consumer --mode live
```
```bash
python -m demo.run_api --open
```

Or one command — `.\run_demo.ps1` on Windows, `./run_demo.sh` on macOS/Linux.

---

## 3. Running modes

`--mode` (or `DEMO_MODE`) selects where alerts come from:

| Mode | Source | Credentials | Use |
|---|---|---|---|
| `live` | Fink Kafka, seeking to the end | yes | **the real demo** |
| `catchup` | Fink Kafka, working through the backlog | yes | filling the database quickly |
| `replay` | Fink Kafka at pinned offsets | yes | **reproducible** — identical every run |
| `rest` | Fink REST API polling | no | while waiting for credentials |
| `offline` | archived `.avro` files on disk | no | no network at all |

`live` seeks to the end of each partition on first connection. That is
deliberate: there can be ~20 000 alerts of backlog, and you do not want to
replay four days of it by accident. Use `catchup --limit N --yes` when you
actually want the backlog.

Useful flags: `--limit N` (stop after N alerts), `--stubs` (fake models, for UI
work), `--verbose`.

---

## 4. The dashboard

### 4.1 Alert stream (`#/`)

The landing page, modelled on the ALeRCE Explorer.

**Left panel — filters.** Object ID, *branch shown* (fused / tabular only /
image only), predicted class, a confidence slider, time window, Kafka topic,
fusion mode, and a "disagreements only" toggle. The branch selector is the
thesis' comparison axis: switch it and the table's class and confidence columns
re-render for that branch alone.

**Main table**, auto-refreshing every 3 seconds:

| Column | Meaning |
|---|---|
| Object ID | ZTF identifier. A badge marks objects from the gold set — red **fitted on** (train/val: not evidence) or blue **held-out test** (out-of-sample). |
| Topic | which Fink filter delivered it |
| Received | how long ago it arrived |
| RA / Dec | sexagesimal |
| Filter, Mag | ZTF-g / r / i, difference-image PSF magnitude |
| Class + Confidence | the selected branch's prediction |
| Agreement | `agree`, `disagree`, or the fusion mode when only one branch ran |
| Latency | broker timestamp → classified. Backlog alerts show `queued Nd` instead, because that number is queue age, not system latency. |

**Statistics panel**: class counts, confidence histogram, latency percentiles,
and a standing caveat that the topic mix imposes a selection prior — live counts
are not an accuracy estimate.

### 4.2 Object detail (click any row)

- **Known classification** callout when we hold an independent label, and
  whether the prediction matches.
- **Per-branch comparison** — tabular, image and fused side by side as bar
  charts. When the branches disagree, a callout says so. When a modality is
  missing, it explains that the surviving branch's calibrated output is used
  unchanged, with no imputation.
- **Light curve** — detections with error bars per filter, upper limits as
  hollow downward triangles, magnitude axis inverted.
- **Pipeline trace** — the alert's journey, stage by stage, with the real
  numbers. Click a stage to expand it; it opens on *Late fusion*, showing the
  exact 6-vector `[log p_tab; log p_img]` fed to the stack.
- **Cutouts** — the science/reference/difference triplet, plus a link to the raw
  `.npy` the model actually consumed.
- **View alert packet** — the raw fields, split into **Instrument (ZTF)** /
  **Broker-derived** / **System**. ALeRCE's Explorer shows one flat table;
  splitting it is where the provenance argument becomes visible.

### 4.3 Methodology (`#/methodology`)

The view for the viva:

1. **Architecture trace** of the most recent alert, end to end.
2. **Branch disagreements**, sharpest contradiction first — where fusion has
   something to resolve. Click through to any object.
3. **Held-out evaluation** — the four scopes, with the significance caveat
   stated: fusion beats the tabular branch by Δmacro-F1 = +0.0071 with a
   bootstrap CI that *includes zero*. The demo argues complementarity, not
   proven superiority.
4. **Models** — the loaded model cards and the shared `split_id`, proving every
   component trained on the same partition.
5. **Provenance** — label sources, what brokers do and do not supply, and how
   that is enforced.

### 4.4 The live indicator (top right)

| Badge | Meaning |
|---|---|
| **LIVE (Kafka)** green | a real push stream is connected |
| CATCHUP / REPLAY / REST fallback / OFFLINE replay | amber — working, but not a live push stream |
| CONSUMER DOWN / STALLED | no heartbeat for 30 s |

Green is reserved for a genuine Kafka stream; a replay can never display it.
Underneath: topic count, consumer lag with a sparkline, time since the last
alert, queue depth, and — deliberately — the **dropped** and **decode failure**
counters. A demo that silently discards data while looking healthy is worse than
one that admits a gap.

It also shows Palomar local time. **A quiet stream in Palomar daylight is
normal, not a fault** — ZTF only observes at night.

---

## 5. How the code works

### 5.1 Two processes, on purpose

```
demo.run_consumer  ──poll──> Kafka ──> classify ──> SQLite (single writer)
                                                       │
demo.run_api       ────────────────────read-only───────┘ ──> dashboard
```

They are separate because the poll loop must never be blocked by an HTTP
handler (lag is a displayed metric), a crash in the web layer must not lose
stream position, and `uvicorn --reload` spawns worker subprocesses — which would
create one Kafka consumer per worker in the same consumer group, producing
duplicate rows.

SQLite runs in WAL mode: one writer, many readers, no server to administer.

### 5.2 The path of one alert

1. **`demo/adapters/fink_kafka.py`** polls Kafka. It drives the underlying
   client directly rather than using `AlertConsumer.poll()`, because that method
   discards the raw message — and with it the partition, offset and broker
   timestamp we need.
2. **`demo/adapters/cutouts.py`** decodes the triplet: gzip → FITS → `float32
   (63, 63)`. Sentinels and NaNs are left intact here; repairing them is the
   image branch's job, so that serving matches training exactly.
3. **`demo/adapters/base.py`** builds the light curve from `prv_candidates`.
   Entries with no magnitude are upper limits, not data points.
4. The result is a **`NormalisedAlert`** (`demo/models.py`) — the single record
   type every source produces, so nothing downstream knows or cares whether the
   alert came from Kafka, REST or a file.
5. **`demo/ingest/consumer_loop.py`** puts it on a bounded queue. If the queue
   is full the newest record is *dropped and counted*, never silently sampled.
6. **`demo/ingest/worker.py`** (a separate thread, owner of the only database
   connection) dequeues and classifies.
7. **`demo/inference/`** runs the branches:
   - `features.py` resolves the 242 features: memory cache → gold cache → disk
     cache → ALeRCE. It carries a circuit breaker, so a blocked network costs
     one request rather than one per alert.
   - `tabular.py` loads `model.txt` via `lightgbm.Booster` (no pickle/sklearn
     version coupling) and applies its OOF-fitted temperature.
   - `image.py` reproduces the training preprocessing exactly: `nan_to_num` →
     sentinel `|v| > 1e30 → 0` → clip to 1–99th percentile → z-score by
     (median, std) → bilinear upsample 63 → 160 → forward pass →
     `softmax(logits / T)`.
   - `fusion.py` computes `softmax(W·[log p_tab; log p_img] + b)` with `W` and
     `b` read from `fusion_card.json` as plain text.
8. **`demo/storage/db.py`** writes the alert, its photometry, the prediction and
   the trace. Stamps go to `.npy` beside the database, not into it.
9. **`demo/api/`** serves it read-only; **`demo/web/`** renders it.

### 5.3 Missing modalities

Fusion never imputes. The stack was fitted only on rows where both branches were
present, so feeding it a uniform vector for an absent branch would apply a
systematic, uncalibrated shift. Instead:

| Situation | `fusion_mode` | Output |
|---|---|---|
| both branches ran | `both` | the learned stack |
| no usable cutouts | `tabular_only` | tabular output, unchanged |
| no features, or fewer than 5 detections | `image_only` | image output, unchanged |
| neither | `none` | stored unclassified, with a reason |

`image_only` is **not** a fault. At first detection there is no light curve to
classify — that is exactly the early-classification regime a multimodal system
exists to cover.

### 5.4 Features arrive late, and that is handled automatically

The tabular branch needs ALeRCE's feature service. Two things routinely leave an
alert without it: the service being unreachable, and — far more often — the
object being too new to have been featurised yet.

Neither requires you to do anything. **The worker retries automatically**: every
60 seconds, while the queue is empty, it re-classifies up to 10 alerts that are
missing their tabular branch. Live alerts always take priority, so this can never
delay the stream. When features become available the alert quietly upgrades to
two-branch fusion, and you will see it in the consumer log:

```
upgraded ZTF26abcfltk to two-branch fusion -> SN (0.99)
```

`scripts/backfill_features.py` does the same thing in bulk, on demand — useful
after a long offline stretch, but not something you need in normal operation.

### 5.5 File map

```
demo/
  config.py            all settings, read from the environment once
  models.py            NormalisedAlert + the instrument/broker/system split
  run_consumer.py      entry point: ingest + classify
  run_api.py           entry point: dashboard + API
  adapters/            Kafka / REST / replay -> NormalisedAlert
  inference/           feature resolution, both branches, fusion, stubs
  ingest/              poll loop, worker thread, health tracking
  storage/             schema, connections, gold-layer bootstrap
  api/                 read-only FastAPI routes
  web/                 dashboard (plain HTML/CSS/JS, no build step)
scripts/
  seed_demo_db.py             populate from the gold layer (offline)
  check_connectivity.py       what is reachable from this machine
  verify_cutouts.py           Phase 0: are cutouts really in the packets
  record_replay_manifest.py   pin offsets for a reproducible replay
  compare_stamp_orientation.py  guards against silently mirrored stamps
  backfill_features.py        bulk feature upgrade (normally automatic)
tests/                        provenance, serving contract, adapters, backfill
```

### 5.6 Configuration

Copy `config/demo.env.example` to `config/demo.env` and edit. Everything has a
working default. The ones that matter:

| Variable | Default | Effect |
|---|---|---|
| `DEMO_MODE` | `live` | ingestion source |
| `DEMO_TOPICS` | six topics | which Fink filters to subscribe to |
| `DEMO_ALERCE_ENABLED` | `1` | set `0` to skip feature lookups entirely |
| `DEMO_USE_STUBS` | `0` | `1` = fake models, for UI work with no artefacts |
| `DEMO_MIN_DETECTIONS` | `5` | below this, no tabular prediction |
| `DEMO_API_PORT` | `8000` | dashboard port |

---

## 6. Running it on another laptop

### 6.1 What you need to copy

| Path | Required? | Notes |
|---|---|---|
| the repository | yes | code |
| `models/` | yes | ~100 MB, gitignored — copy manually |
| `data/gold/` | for seeding | `gold_stamps.npz` is large; `_cache_features/` is small and valuable |
| `data/demo/` | optional | copy `demo.db` + `stamps/` to move a populated demo |
| `~/.finkclient/ztf_credentials.yml` | for Kafka | or just re-run `fink_client_register` |
| `config/replay_manifest.json` | for replay mode | committed to git |

Nothing in the code uses absolute paths — everything resolves against the
repository root — so the checkout can live anywhere.

### 6.2 Setup

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements-demo.txt
python scripts/seed_demo_db.py --n 250
python -m demo.run_api --open
```

If the dashboard shows alerts, everything works. Then re-register Fink
credentials and try `--mode live`.

### 6.3 Does it work on macOS?

**Yes**, including Apple Silicon. The code is platform-neutral: `pathlib`
throughout, no shell-outs, no Windows APIs. Four things to know:

1. **LightGBM needs OpenMP on macOS.** This is the one real gotcha — `import
   lightgbm` fails without it:
   ```bash
   brew install libomp
   ```
2. **Use `run_demo.sh`**, not `run_demo.ps1`:
   ```bash
   chmod +x run_demo.sh && ./run_demo.sh --mode offline --seed
   ```
3. **`python3`, not `python`** on most macOS setups. The script honours a
   `PYTHON` environment variable if yours is elsewhere.
4. **PyTorch on Apple Silicon** installs a native arm64 build. Inference is CPU
   and takes ~4 ms per stamp either way, so there is nothing to configure.

`tzdata` is Windows-only in the requirements (macOS and Linux have a system
timezone database). The timezone lookup degrades gracefully in any case.

Everything else — `confluent-kafka`, `fastavro`, `astropy`, `fastapi` — ships
macOS wheels for both Intel and Apple Silicon.

### 6.4 Linux

Same as macOS without the `libomp` step — LightGBM's Linux wheels bundle it.

---

## 7. Troubleshooting

**Dashboard says "CONSUMER DOWN" but the consumer is running.**
The consumer heartbeats every ~2 s, even when idle; 30 s of silence marks it
down. If it is genuinely running, check its terminal for a traceback. "CONSUMER
STALLED" means the process exists but stopped heartbeating.

**No alerts arriving in `live` mode.**
Probably normal. Check Palomar local time in the indicator — ZTF only observes
at night. To confirm the connection works, use `--mode catchup --limit 20 --yes`.

**Everything is `image_only`.**
The feature service is unreachable, or the objects are too new to have been
featurised. Run `python scripts/check_connectivity.py`. Either way the worker
keeps retrying, and stored alerts upgrade themselves once features appear.

**`sasl.username and sasl.password must be set`.**
The credentials file must have `password: null`. Re-run `fink_client_register`
with `-survey ztf` and no `-password`.

**Consumer connects to `localhost:9093`.**
Its `bootstrap.servers` was not set. Re-register; the credentials file should
contain `servers: kafka-ztf.fink-broker.org:24499`.

**The UI does not reflect an edit.**
It should — assets are versioned by file mtime. If not, hard-reload.

**`ModuleNotFoundError: lightgbm` on macOS.** `brew install libomp`.

**Start completely fresh.**
```bash
python scripts/seed_demo_db.py --n 250 --reset
```

---

## 8. Checks worth running

```bash
python tests/test_provenance.py        # broker classifications cannot reach a model
python tests/test_serving_contract.py  # serving reproduces the trained metrics
python tests/test_adapters.py          # silent-failure invariants
python tests/test_backfill.py          # the image-only -> fusion upgrade path
```

`test_serving_contract.py` is the one to run after touching anything in
`demo/inference/`: it re-derives the fusion card's held-out macro-F1 (0.9528)
through the serving code and fails if it drifts by more than 1e-4.
