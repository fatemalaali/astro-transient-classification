-- Demo database schema. See docs/demo-plan.md section 6.5.
--
-- SQLite in WAL mode: exactly one writer (the consumer process) and N readers
-- (the API), which is the only concurrency property this demo needs.
--
-- Deviation from the plan document: the Kafka coordinate columns are named
-- `kafka_partition` / `kafka_offset` rather than `partition` / `offset`.
-- OFFSET is a reserved word in SQLite and PARTITION is reserved in window
-- functions, so the unprefixed names would need quoting in every query.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- --------------------------------------------------------------------------
-- alerts: one row per alert packet, whatever the source
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    candid              INTEGER PRIMARY KEY,   -- ZTF candid, unique per alert
    object_id           TEXT    NOT NULL,
    source              TEXT    NOT NULL,      -- fink_kafka|fink_rest|alerce_rest|replay
    topic               TEXT,
    kafka_partition     INTEGER,
    kafka_offset        INTEGER,

    jd                  REAL    NOT NULL,      -- exposure mid-point, Julian Date
    mjd                 REAL    NOT NULL,
    emitted_utc         TEXT    NOT NULL,      -- ISO-8601 UTC, derived from jd
    kafka_ts_utc        TEXT,                  -- Kafka broker timestamp
    broker_ingest_utc   TEXT,                  -- packet 'timestamp' (broker-derived)
    received_utc        TEXT    NOT NULL,      -- our clock at poll

    ra                  REAL    NOT NULL,      -- degrees, ICRS
    dec                 REAL    NOT NULL,      -- degrees, ICRS
    fid                 INTEGER NOT NULL,      -- 1=g 2=r 3=i
    magpsf              REAL,
    sigmapsf            REAL,
    diffmaglim          REAL,
    isdiffpos           TEXT,

    distnr              REAL,
    magnr               REAL,
    sgscore1            REAL,
    distpsnr1           REAL,
    neargaia            REAL,
    rb                  REAL,                  -- display only; bogus is out of scope
    drb                 REAL,

    ndethist            INTEGER,
    n_det               INTEGER NOT NULL DEFAULT 0,
    n_nondet            INTEGER NOT NULL DEFAULT 0,

    cutout_status       TEXT    NOT NULL,      -- ok|partial|missing|decode_error
    stamp_path          TEXT,
    raw_packet_ref      TEXT,

    -- Every broker classification lives here, and ONLY here. No column in this
    -- schema outside this blob carries a broker-derived prediction.
    broker_meta_json    TEXT,

    UNIQUE (topic, kafka_partition, kafka_offset)
);

CREATE INDEX IF NOT EXISTS ix_alerts_received ON alerts (received_utc DESC);
CREATE INDEX IF NOT EXISTS ix_alerts_object   ON alerts (object_id);
CREATE INDEX IF NOT EXISTS ix_alerts_topic    ON alerts (topic, received_utc DESC);

-- --------------------------------------------------------------------------
-- predictions: one row per classified alert
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    candid              INTEGER PRIMARY KEY
                        REFERENCES alerts (candid) ON DELETE CASCADE,
    status              TEXT    NOT NULL,      -- ok|unclassified|error
    status_reason       TEXT,
    fusion_mode         TEXT    NOT NULL,      -- both|tabular_only|image_only|none

    p_tab_sn REAL, p_tab_agn REAL, p_tab_vs REAL,
    p_img_sn REAL, p_img_agn REAL, p_img_vs REAL,
    p_fused_sn REAL, p_fused_agn REAL, p_fused_vs REAL,

    predicted_class     TEXT,                  -- SN|AGN|VS
    confidence          REAL,                  -- max(p_fused)
    branch_disagree     INTEGER NOT NULL DEFAULT 0,
    fusion_flips        INTEGER NOT NULL DEFAULT 0,

    feature_provenance  TEXT,                  -- gold_cache|disk_cache|alerce_live|unavailable
    n_features_present  INTEGER,

    t_feature_ms REAL, t_stamp_ms REAL, t_tab_ms REAL, t_img_ms REAL, t_fuse_ms REAL,
    t_pipeline_ms REAL,
    t_broker_to_classified_ms REAL,
    t_emitted_to_classified_s REAL,

    model_versions_json TEXT,
    trace_json          TEXT,                  -- the methodology trace payload
    split_id            TEXT,                  -- ties a live row to the trained partition
    created_utc         TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_pred_class ON predictions (predicted_class, confidence DESC);
CREATE INDEX IF NOT EXISTS ix_pred_disagree ON predictions (branch_disagree)
    WHERE branch_disagree = 1;
CREATE INDEX IF NOT EXISTS ix_pred_mode ON predictions (fusion_mode);

-- --------------------------------------------------------------------------
-- known_labels: ground truth from OUR OWN label sources only.
-- TNS, BTS, Chen+2020, Milliquas, SDSS DR16Q. Never a broker classifier.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS known_labels (
    object_id       TEXT PRIMARY KEY,
    coarse          TEXT,                      -- SN|AGN|VS
    fine            TEXT,
    plasticc_class  TEXT,                      -- carried through untouched for RQ3
    label_source    TEXT NOT NULL,             -- bts|tns|chen_vs|sdss|milliquas
    in_training_set INTEGER NOT NULL DEFAULT 0,
    training_split  TEXT                       -- train|val|test|NULL
);

CREATE INDEX IF NOT EXISTS ix_labels_training ON known_labels (in_training_set);

-- --------------------------------------------------------------------------
-- photometry: light-curve points for the object-detail plot
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS photometry (
    candid      INTEGER NOT NULL REFERENCES alerts (candid) ON DELETE CASCADE,
    object_id   TEXT    NOT NULL,
    jd          REAL    NOT NULL,
    fid         INTEGER NOT NULL,
    magpsf      REAL,
    sigmapsf    REAL,
    diffmaglim  REAL,
    kind        TEXT    NOT NULL,              -- detection|nondetection
    PRIMARY KEY (candid, jd, fid, kind)
);

CREATE INDEX IF NOT EXISTS ix_phot_object ON photometry (object_id, jd);

-- --------------------------------------------------------------------------
-- stream_health: one row per poll cycle, for the live indicator
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stream_health (
    ts_utc          TEXT PRIMARY KEY,
    mode            TEXT    NOT NULL,          -- live|replay|catchup|rest|offline
    connected       INTEGER NOT NULL,
    is_live_stream  INTEGER NOT NULL DEFAULT 0,
    topics_json     TEXT,
    lag_json        TEXT,
    committed_json  TEXT,
    last_alert_utc  TEXT,
    queue_depth     INTEGER NOT NULL DEFAULT 0,
    dropped_total   INTEGER NOT NULL DEFAULT 0,
    decode_failures INTEGER NOT NULL DEFAULT 0,
    consumed_total  INTEGER NOT NULL DEFAULT 0,
    consumer_pid    INTEGER,
    error           TEXT,
    -- Reachability of the ALeRCE feature service as seen by the *ingest*
    -- process. This is what decides whether live alerts get a tabular branch,
    -- so it belongs next to the stream state rather than in a separate table.
    alerce_json     TEXT
);

-- --------------------------------------------------------------------------
-- eval_summary: static held-out metrics, loaded from the model cards
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_summary (
    scope               TEXT PRIMARY KEY,      -- tabular|image|fused|equal_weight
    label               TEXT,
    macro_f1            REAL,
    balanced_accuracy   REAL,
    accuracy            REAL,
    log_loss            REAL,
    fold                TEXT,
    split_id            TEXT
);

-- --------------------------------------------------------------------------
-- meta: schema version and bootstrap bookkeeping
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS meta (
    key     TEXT PRIMARY KEY,
    value   TEXT
);
