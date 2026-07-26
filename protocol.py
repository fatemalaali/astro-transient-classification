"""Shared training / evaluation protocol for the astro-transient-classification thesis.

This module is the **single source of truth** for the one protocol every branch
notebook obeys, in this exact order:

1. **Fit** candidate models on the **training** fold.
2. **Tune** hyper-parameters on the **validation** fold (Optuna/TPE).
3. **Select** the winning candidate on **validation** macro-F1 — never on test
   (:func:`select_winner`).
4. **Generate out-of-fold (OOF) predictions** for the winner by forward-chaining
   inside the training fold (:func:`forward_chain_oof`).
5. **Fit calibration temperature and the fusion meta-learner on those OOF
   predictions** (:func:`fit_temperature`) — not on validation, not in-sample.
6. **Refit** the winner on **train + validation** at the tuned config — the
   deployed/reported model.
7. **Read the test fold exactly once**, at the very end.

The three folds map to three jobs: *training fits, validation decides, test
grades (once).*

Design note: everything that a plain-`sklearn` environment cannot provide (torch,
lightgbm, xgboost, imbalanced-learn) is imported **lazily inside the function that
needs it**, so ``import protocol`` succeeds in every notebook environment
(the CPU tabular env and the ``cv_gpu`` CUDA env alike) with no notebook
dependency of its own.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SEED = 42
N_BLOCKS = 5
COARSE_CLASSES = ("SN", "AGN", "VS")


# --------------------------------------------------------------------------- #
# split identity
# --------------------------------------------------------------------------- #
def load_split(gold_dir) -> pd.DataFrame:
    """Return ``oid, split, firstmjd`` for the gold set."""
    gold_dir = Path(gold_dir)
    spl = pd.read_parquet(gold_dir / "gold_splits.parquet")
    cols = [c for c in ("oid", "split", "firstmjd") if c in spl.columns]
    return spl[cols].copy()


def split_id(split_df: pd.DataFrame) -> str:
    """Canonical identity = short hash of the sorted ``oid -> split`` assignment.

    Order-independent and environment-independent by construction (it hashes a
    sorted ``oid\\tsplit`` text payload, not a pandas object hash), so two models
    trained on the same partition always produce the *same* string — unlike the
    old per-notebook ``pd.util.hash_pandas_object`` hashes, which differed
    cosmetically with ``index=`` and row order. Used to assert every model was
    trained on the same partition.
    """
    d = split_df[["oid", "split"]].copy()
    d["oid"] = d["oid"].astype(str)
    d["split"] = d["split"].astype(str)
    d = d.sort_values("oid").reset_index(drop=True)
    payload = "\n".join(f"{o}\t{s}" for o, s in zip(d["oid"], d["split"]))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def assert_same_split(cards: list[dict]) -> str:
    """Verify split identity across model cards by the canonical :func:`split_id`.

    Each card must carry a ``split_id`` (the canonical hash of its full
    ``oid -> split`` map). Equal ``split_id`` is equivalent to identical OID sets
    per fold — that is exactly what the canonical hash encodes — so comparing the
    canonical strings *is* the OID-set comparison, without the cosmetic
    disagreements the old ``split_hash`` strings suffered from. Returns the shared
    id on success; raises ``AssertionError`` otherwise.
    """
    ids = [c.get("split_id") for c in cards]
    missing = [i for i, v in enumerate(ids) if v is None]
    assert not missing, f"model cards missing canonical split_id: card indices {missing}"
    uniq = set(ids)
    assert len(uniq) == 1, f"split identity mismatch across model cards: {sorted(uniq)}"
    return ids[0]


# --------------------------------------------------------------------------- #
# selection  —  THE selection rule, used everywhere
# --------------------------------------------------------------------------- #
def select_winner(val_metrics: pd.DataFrame, metric: str = "macro_f1"):
    """THE selection rule: argmax over the **validation** metric frame.

    One place, used by every notebook. ``val_metrics`` is indexed by model
    name/label with a column ``metric``. Returns the index label of the winner.
    Selecting on test anywhere is a protocol violation — this function only ever
    sees a validation frame.
    """
    if metric not in val_metrics.columns:
        raise KeyError(f"selection metric {metric!r} not in validation frame columns "
                       f"{list(val_metrics.columns)}")
    return val_metrics[metric].idxmax()


# --------------------------------------------------------------------------- #
# forward-chaining OOF  (generic over branch type)
# --------------------------------------------------------------------------- #
def forward_chain_oof(fit_predict, train_idx, firstmjd, n_blocks: int = N_BLOCKS):
    """Time-ordered forward-chaining out-of-fold predictions inside the train fold.

    Sort ``train_idx`` by ``firstmjd``; cut into ``n_blocks`` contiguous,
    time-ordered blocks. For ``r`` in ``0 .. n_blocks-2``: train on blocks
    ``0..r`` and predict block ``r+1``. Block 0 is never predicted.

    Parameters
    ----------
    fit_predict : callable
        ``fit_predict(train_subset_idx, predict_idx) -> proba`` where the index
        arrays are **global** positions (into the full-length arrays the caller
        works with) and the returned array has shape ``(len(predict_idx), n_classes)``.
        It MUST use the FIXED tuned rounds/epochs — never early-stop on the block
        it is about to predict.
    train_idx : array-like of int
        Global positions of the training-fold objects.
    firstmjd : array-like
        Full-length ``firstmjd`` array (indexed by the same global positions).
    n_blocks : int
        Number of contiguous temporal blocks (default :data:`N_BLOCKS`).

    Returns
    -------
    oof_proba : np.ndarray  ``(n_predicted, n_classes)``
    oof_idx : np.ndarray    global positions the rows of ``oof_proba`` correspond to
        (blocks ``1 .. n_blocks-1``, concatenated in time order). Map these to
        oids with ``oids[oof_idx]`` before saving.
    """
    train_idx = np.asarray(train_idx)
    firstmjd = np.asarray(firstmjd)
    order = np.argsort(firstmjd[train_idx], kind="stable")
    ordered = train_idx[order]
    blocks = np.array_split(ordered, n_blocks)

    proba_parts, idx_parts = [], []
    for r in range(n_blocks - 1):
        tr = np.concatenate(blocks[: r + 1])
        pr = blocks[r + 1]
        if len(pr) == 0:
            continue
        proba = np.asarray(fit_predict(tr, pr), dtype=np.float64)
        assert proba.shape[0] == len(pr), (
            f"fit_predict returned {proba.shape[0]} rows for a block of {len(pr)}")
        proba_parts.append(proba)
        idx_parts.append(pr)

    oof_proba = np.vstack(proba_parts)
    oof_idx = np.concatenate(idx_parts)
    return oof_proba, oof_idx


# --------------------------------------------------------------------------- #
# probability helpers + temperature scaling
# --------------------------------------------------------------------------- #
_LOG_EPS = 1e-12


def as_logits(proba) -> np.ndarray:
    """Log-probabilities, usable as logits for a probabilistic classifier.

    Temperature scaling on ``log p`` is identical to scaling on the raw logits,
    because softmax is invariant to per-row additive constants — so a single
    :func:`fit_temperature` serves both the tabular (probabilistic) and CNN
    (logit) branches when fed ``as_logits(proba)``.
    """
    return np.log(np.clip(np.asarray(proba, dtype=np.float64), _LOG_EPS, 1.0))


def softmax(z, T: float = 1.0) -> np.ndarray:
    """Numerically-stable softmax with temperature ``T`` (row-wise)."""
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class TemperatureScaler:
    """Single-scalar temperature scaling (Guo et al., 2017), fitted on OOF logits.

    Kept torch-free so it is usable in every environment; :meth:`fit` minimises
    the OOF negative log-likelihood over a single ``T`` and :meth:`transform`
    returns calibrated probabilities.
    """

    def __init__(self, T: float = 1.0):
        self.T = float(T)

    def fit(self, logits, y):
        self.T = fit_temperature(logits, y)
        return self

    def transform(self, logits) -> np.ndarray:
        return softmax(logits, T=self.T)


def fit_temperature(logits, y) -> float:
    """Single scalar ``T`` minimising NLL on OOF predictions.

    For probabilistic branches (LightGBM) pass ``as_logits(proba)``; for CNN
    branches pass the raw logits (equivalent — see :func:`as_logits`).
    """
    from scipy.optimize import minimize

    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y)

    def nll(log_t):
        z = logits / np.exp(log_t[0])
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean()

    res = minimize(nll, x0=np.array([0.0]), method="L-BFGS-B")
    return float(np.exp(res.x[0]))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def macro_f1(y, proba) -> float:
    """Macro-averaged F1 over all classes present in ``proba``'s width."""
    from sklearn.metrics import f1_score

    y = np.asarray(y)
    y_pred = np.asarray(proba).argmax(1)
    labels = list(range(np.asarray(proba).shape[1]))
    return float(f1_score(y, y_pred, average="macro", labels=labels, zero_division=0))


def expected_calibration_error(proba, y, n_bins: int = 15) -> float:
    """Expected calibration error over ``n_bins`` equal-width confidence bins."""
    proba = np.asarray(proba)
    y = np.asarray(y)
    conf = proba.max(1)
    pred = proba.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def bootstrap_ci(metric_fn, y, proba, n: int = 10_000, seed: int = SEED,
                 alpha: float = 0.05):
    """Percentile bootstrap CI for ``metric_fn(y, proba)`` over the given objects.

    Returns ``(point, lo, hi)``. ``metric_fn`` takes ``(y, proba)`` and returns a
    scalar (e.g. :func:`macro_f1`).
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    rng = np.random.default_rng(seed)
    n_obj = len(y)
    point = float(metric_fn(y, proba))
    draws = np.empty(n)
    for i in range(n):
        s = rng.integers(0, n_obj, n_obj)
        draws[i] = metric_fn(y[s], proba[s])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar's test on two boolean correctness vectors.

    Returns discordant counts, the continuity-corrected chi-square statistic and
    the two-sided exact-binomial p-value.
    """
    a = np.asarray(correct_a).astype(bool)
    b = np.asarray(correct_b).astype(bool)
    n01 = int((~a & b).sum())   # a wrong, b right
    n10 = int((a & ~b).sum())   # a right, b wrong
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    try:
        from scipy.stats import binomtest
        p = float(binomtest(min(n01, n10), n, 0.5).pvalue)
    except Exception:  # pragma: no cover - scipy always present with sklearn
        from scipy.stats import binom
        k = min(n01, n10)
        p = float(min(1.0, 2.0 * binom.cdf(k, n, 0.5)))
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": p}


# --------------------------------------------------------------------------- #
# shared tabular model helpers  (used by the lc_* branches)
# --------------------------------------------------------------------------- #
def _impute_steps(scale: bool):
    """Median impute + missing-indicator, optionally followed by standard scaling."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


def make_model(algo: str, params: dict | None = None, n_classes: int = 3,
               seed: int = SEED):
    """Build one class-balanced tabular learner (shared by the lc_* notebooks).

    Trees (LightGBM / XGBoost) consume NaNs natively and use class weights; BRF /
    MLP are wrapped in an impute (+scale, +oversample for the MLP) pipeline.
    ``params`` (from Optuna) overrides the defaults; for the pipeline models
    Optuna passes ``clf__``-prefixed keys.
    """
    params = dict(params or {})
    if algo == "lightgbm":
        from lightgbm import LGBMClassifier
        base = dict(n_estimators=600, learning_rate=0.05, num_leaves=31,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    reg_lambda=1.0, min_child_samples=20, objective="multiclass",
                    class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1)
        base.update(params)
        return LGBMClassifier(**base)

    if algo == "xgboost":
        from xgboost import XGBClassifier
        base = dict(n_estimators=600, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    min_child_weight=1.0, gamma=0.0, objective="multi:softprob",
                    num_class=n_classes, tree_method="hist", eval_metric="mlogloss",
                    random_state=seed, n_jobs=-1)
        base.update(params)
        return XGBClassifier(**base)

    if algo == "brf":
        from imblearn.ensemble import BalancedRandomForestClassifier
        from sklearn.pipeline import Pipeline
        base = dict(n_estimators=400, max_features="sqrt", min_samples_leaf=1,
                    criterion="gini", sampling_strategy="all", replacement=True,
                    bootstrap=False, random_state=seed, n_jobs=-1)
        pipe = Pipeline(_impute_steps(scale=False) +
                        [("clf", BalancedRandomForestClassifier(**base))])
        if params:
            pipe.set_params(**params)
        return pipe

    if algo == "mlp":
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.neural_network import MLPClassifier
        base = dict(hidden_layer_sizes=(128, 64), alpha=1e-4, learning_rate_init=1e-3,
                    batch_size=256, max_iter=300, early_stopping=True,
                    n_iter_no_change=15, validation_fraction=0.1, random_state=seed)
        pipe = ImbPipeline(_impute_steps(scale=True) +
                           [("ros", RandomOverSampler(random_state=seed)),
                            ("clf", MLPClassifier(**base))])
        if params:
            pipe.set_params(**params)
        return pipe

    raise ValueError(algo)


def class_sample_weights(y):
    """Per-sample weights from balanced class weights (for XGBoost's ``fit``)."""
    from sklearn.utils.class_weight import compute_class_weight

    y = np.asarray(y)
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    lut = {int(c): float(wi) for c, wi in zip(classes, w)}
    return np.array([lut[int(v)] for v in y])


def fit_model(model, algo: str, Xtr, ytr, Xval=None, yval=None):
    """Fit a tabular model, using the temporal val set for early stopping where
    supported. With ``Xval is None`` no early stopping happens — this is the mode
    the forward-chaining OOF blocks and the train+val refit use.
    """
    if algo == "lightgbm":
        from lightgbm import early_stopping, log_evaluation
        cbs, es = [], ([(Xval, yval)] if Xval is not None else None)
        if es is not None:
            cbs = [early_stopping(50, verbose=False), log_evaluation(0)]
        model.fit(Xtr, ytr, eval_set=es, callbacks=cbs)
    elif algo == "xgboost":
        model.set_params(early_stopping_rounds=50 if Xval is not None else None)
        es = [(Xval, yval)] if Xval is not None else None
        model.fit(Xtr, ytr, sample_weight=class_sample_weights(ytr),
                  eval_set=es, verbose=False)
    else:                                            # BRF / MLP pipelines
        model.fit(Xtr, ytr)
    return model
