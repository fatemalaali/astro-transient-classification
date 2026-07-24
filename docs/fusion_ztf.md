# Late-Fusion Classifier — ZTF Gold (companion write-up)

Companion to `fusion_ztf.ipynb`. This document records *what* was done and *why*, in a
form ready to lift into the methodology and results chapters. It uses UK spelling and
Harvard author–year citations; a reference list is at the end. All figures referenced
live in `figures/fusion/` (300-dpi PNG + vector PDF); the saved meta-learner lives in
`models/fusion/logreg_stack/`.

This is the third and final classifier branch for the ZTF gold dataset. It consumes the
two unimodal branches — `lc_classifier_ztf.ipynb` (tabular light-curve features) and
`stamp_classifier_ztf.ipynb` (image stamps) — and learns a decision-level meta-learner
over their probability vectors. It also documents, in §3, a methodological problem in the
two branch pipelines that had to be resolved before any fusion could be fitted at all.

---

## 1. Data and the temporal protocol

The fusion branch introduces no new data. It reuses the gold layer built by
`build_dataset.ipynb`, and in particular the single persisted temporal split in
`gold_splits.parquet`: objects are ordered by `firstmjd` and cut 70 / 15 / 15 into
train (8 278), validation (1 773) and test (1 775). The split is object-disjoint and
strictly time-ordered, so the test fold is "future" relative to training, mimicking
deployment. It is deliberately *not* stratified, so class balance drifts across folds
by construction.

**Object-level correspondence.** Both branches are keyed on `oid` (the ZTF object id),
with exactly one row per object in every gold table — one feature vector and one
science/reference/difference stamp triplet each. Tabular and image samples are therefore
1:1 with no alert-level aggregation, which is what makes decision-level fusion clean here:
the meta-learner sees two predictions about the *same* object rather than two predictions
about loosely-associated events.

**Compatibility verification.** Fusion is only meaningful if both branches saw the same
objects in the same folds. The notebook verifies this on **oid sets, per split**, and
computes its own canonical hash (`50591cf87b04`, sorted by `oid`, `index=False`) that
both branches would agree on.

The stored `split_hash` strings in the two model cards *disagree* — `65e15eed88f2` for the
light-curve branch against `5ade91048434` for the stamp branch — and `stamp_classifier_ztf.md`
§7 claims these are cross-checked to confirm fusion compatibility. Taken literally that
check fails. The difference is a hashing artefact, not a real one: the light-curve notebook
hashes with `index=True` over its merge-ordered frame while the stamp notebook hashes with
`index=False` over the NPZ-ordered frame. The underlying folds are identical. Comparing the
stored strings would raise a false alarm and comparing them successfully would be luck, so
the fusion notebook compares the sets themselves.

---

## 2. Branch selection

The model carried into fusion from each branch is the documented winner of that branch:
**LightGBM** for the tabular modality and **EfficientNet-B0** for the image modality. No
re-selection is performed in the fusion notebook. Two caveats belong in the record.

**The winners were selected on the test fold, not the validation fold.** The stated
protocol is that the candidate with the highest validation macro-F1 is carried forward,
leaving the test fold untouched until final reporting. Both branch notebooks instead
compared candidates on test: `figures/lc/ztf/verdict.csv` assigns `is_winner` on test
macro-F1, and `stamp_classifier_ztf.ipynb` prints "Best ZTF stamp model by test macro-F1".
A model chosen as the maximum of three or four on a given fold carries the optimism of
that choice, so both branch test figures are mildly optimistic as estimates of unseen
performance. For the tabular branch the two rules agree (LightGBM wins on both). For the
image branch they disagree:

| Model | val macro-F1 (clean, train-only) | test macro-F1 |
|---|---|---|
| ResNet-18 | **0.780** | 0.7508 |
| EfficientNet-B0 | 0.757 | **0.7676** |

The validation rule would have selected ResNet-18. The decision taken here is to keep
EfficientNet-B0 for continuity with the published branch results and to disclose the
departure rather than silently correct it. This is a limitation, recorded in §8.

The clean validation figures above are the best trial values recorded by each architecture's
Optuna study, which trained on the training fold only. When §3 refits the *same*
EfficientNet-B0 configuration on the training fold it obtains 0.7424 rather than 0.757. The
0.015 discrepancy is ordinary run-to-run variance — stochastic augmentation, sampler order
and cuDNN kernel selection all differ between runs, and the early-stopping epoch moves with
them. It is a useful scale reference: differences between architectures smaller than roughly
0.02 validation macro-F1 should not be treated as real, which is another reason the
0.780-against-0.757 ordering above is not decisive.

**The tabular margin is not meaningful.** LightGBM leads XGBoost by 0.0003 validation
macro-F1 (0.9346 against 0.9343). The choice between two boosters performing this
similarly is effectively arbitrary, and no conclusion in this document should be read as
depending on it.

**A further caveat on the underlying searches.** The stamp branch's Optuna studies
terminated early — the notebook output records `3/20 complete` for ResNet-18 and
`5/20 complete` for EfficientNet-B0. Both architectures were therefore selected on a
handful of hyper-parameter trials, so neither the 0.780 nor the 0.757 figure is a
well-converged estimate of what that architecture can do.

---

## 3. The validation-fold leakage problem

This section is the central methodological content of the fusion branch. It documents a
problem in the two branch pipelines that makes the fusion protocol, as originally
specified, impossible to execute against the saved artefacts — and the minimal remedy
adopted.

### 3.1 What the branch notebooks saved

Following stacked generalisation (Wolpert, 1992), the meta-learner must be fitted on
**held-out** branch predictions, so that no branch can pass memorised training labels into
the fusion stage. The specification says these come from the validation fold.

Both branch notebooks, however, refit their final model on **train + val** before saving:

- `lc_classifier_ztf.ipynb` cell 47 — `final = fit_model(make_model(algo, best_params[algo]), algo, Xtrval, ytrval)`
- `stamp_classifier_ztf.ipynb` cell 43 — `state, _, _, _ = train_model(arch, params, TRAINVAL_IDX, ...)`

This is orthodox practice for a *deployed* model: once hyper-parameters are fixed, using
all labelled data is the right choice. But it means the validation fold is training data
for both saved models, so those models cannot supply held-out predictions about it.

### 3.2 How much it matters

Scoring the deployed artefacts on every fold (`01_leakage_audit`, `leakage_audit.csv`):

| Branch | train | val | test | val mean max-*p* | val log loss |
|---|---|---|---|---|---|
| LightGBM (LC) | 1.0000 | **1.0000** | 0.9457 | **0.9999** | 0.0001 |
| EfficientNet-B0 (stamp) | 0.9062 | 0.8814 | 0.7676 | 0.9586 | 0.1084 |

Both test figures reproduce their branch cards exactly (0.9457140305 and 0.7676485433),
which is the check that the artefacts are being loaded and pre-processed correctly.

Reaching that exactness on the image branch required disabling `cudnn.benchmark`, which the
branch notebooks enable for training throughput. With it on, cuDNN selects convolution
algorithms non-deterministically and repeat inference runs of the same frozen network differ
by around 0.001 macro-F1 — enough to make the audit table irreproducible at four decimal
places. The fusion notebook therefore sets `benchmark = False` and
`use_deterministic_algorithms(True, warn_only=True)`. The cost is irrelevant here because the
only training is the cached refit of §3.

The tabular branch is **fully degenerate on the validation fold**: a 1 100-tree booster
reproduces those 1 773 labels essentially exactly, emitting one-hot vectors that are always
correct. Two consequences follow, and both are fatal rather than merely inconvenient.

**The meta-learner degenerates.** Fitting the proposed multinomial logistic regression on
these validation probabilities and sweeping the regularisation strength:

| C | Σ\|W_tab\| | Σ\|W_img\| | ratio | val macro-F1 |
|---|---|---|---|---|
| 0.01 | 3.80 | 3.10 | 1.23 | 0.6534 |
| 0.10 | 9.00 | 5.84 | 1.54 | 0.9942 |
| 1.00 | 15.33 | 7.69 | 1.99 | **1.0000** |
| 10.0 | 21.13 | 9.30 | 2.27 | **1.0000** |
| 100 | 25.37 | 9.98 | 2.54 | **1.0000** |

The objective saturates. The meta-learner learns "copy the tabular vector" and there is no
regularisation strength at which the fold carries information about relative modality
trust. This is not an optimistic fit that would merely overstate the fusion gain; it is a
fit that contains no signal at all.

**Temperature scaling the tabular branch is unimplementable.** The negative log-likelihood
of perfectly-memorised data is minimised as `T → 0`, so the LBFGS fit diverges rather than
converging on a meaningful scalar.

**A related trap in the published numbers.** The two branches' card `val_metrics` are not
comparable as printed. The light-curve card's value (0.9346) comes from a *separate*
train-only model built purely to score validation (`vmodel` in cell 47) and is clean; the
stamp card's (0.8814) is the train+val model scored on val, and is in-sample. Comparing
them directly — for instance to argue that the image branch suffers more temporal drift —
is not a like-for-like comparison and would support the wrong conclusion.

### 3.3 The remedy

The same two tuned configurations are reloaded from their persisted `best_params.json` and
**refit on the training fold alone**. No re-tuning and no re-selection occurs: the carried
models are still LightGBM and EfficientNet-B0 with the same hyper-parameters. The only
purpose is to obtain honest held-out validation predictions.

One residual dependence remains and is stated rather than hidden: the number of boosting
rounds (tabular) and the stopping epoch (image) are still chosen by early stopping on the
validation fold. That is a single scalar per branch, and it is exactly how the light-curve
branch computed the clean validation metrics in its own card, so the two remain comparable.

Both branches' probability matrices are cached to `models/fusion/_probs_cache.npz`, keyed on
a hash of the two parameter files plus the split, so re-running the notebook skips both
refits.

### 3.4 Which models are evaluated on test

Two provenances are reported, because they answer different questions.

- **Option A (headline, `base_model_provenance: train_only_refit`).** The train-only models
  are used for both fitting the fusion and evaluating it. Fully leak-free and internally
  coherent: the fusion weights were fitted for exactly these base models.
- **Option B (robustness row, `deploy_refit`).** The fitted fusion is applied to the
  *deployed* train+val models' test probabilities. These are the models the branch
  documents report and the alert system would ship, so this row keeps the fusion result
  comparable with the published branch numbers — at the cost that the fusion parameters
  were fitted for differently-trained base models.

The gap between the two is itself a reportable quantity and appears in §6.

---

## 4. Calibration

Because the fusion stage combines probabilities, both branches are calibrated before
stacking. An uncalibrated branch would silently dominate the fusion weights regardless of
its true reliability, and convolutional networks in particular tend to be over-confident
(Guo et al., 2017).

Each branch is calibrated by **temperature scaling**: a single scalar `T` dividing the
logits before the softmax, `p = softmax(z / T)`, fitted on the validation fold by
minimising the negative log-likelihood. For the tabular branch, which emits probabilities
rather than logits, the log-probabilities serve as logits — softmax of `log p / T` is the
standard temperature family for a probabilistic classifier.

Note that calibrating the tabular branch is *only possible* because §3 restored a
non-degenerate validation fold. Against the deployed booster the fit diverges.

Calibration quality is checked with reliability diagrams and the expected calibration
error before and after scaling (`02_reliability`, `calibration.csv`), on both the
validation and test folds. Because temperature scaling is a monotone transform of each
branch's logits, it cannot change any branch's own argmax predictions or its macro-F1 — it
changes only how much weight those predictions carry once combined.

The temperature fitted here for the image branch differs from the value stored in its
model card, because the card's was fitted against the train+val model on data that model
had memorised.

---

## 5. Fusion design

### 5.1 Why late fusion

Fusion is performed at the decision level: the meta-learner consumes the concatenation of
the two branch probability vectors, `z = [p_tab ; p_img]`, and returns the final
distribution,

<!-- eq:fusion -->
> `p_fused = softmax(W z + b)`,  `ŷ = argmax_c p_fused`

with `W` (3 × 6) and `b` (3) estimated by regularised cross-entropy. Six inputs for the
coarse task keeps the fitted weights small enough to read directly as the relative trust
placed in each modality, per class.

Late fusion is preferred over early (feature-concatenation) and intermediate
(joint-embedding) fusion for four reasons that follow from the research design. First,
keeping the branches modality-pure is what makes the unimodal-versus-multimodal ablation
well-posed: the same two branch models serve both as the baselines and as the inputs to
fusion, so the only added quantity is the fusion mechanism itself. Second, each modality is
served by the architecture best suited to it — gradient-boosted trees for tabular features,
a convolutional network for images — rather than being forced into a single shared model.
Third, probability-level combination degrades gracefully when one modality is weak or
absent, a realistic condition in a live alert stream. Fourth, the meta-learner is negligible
in cost, preserving the real-time budget.

### 5.2 The three rungs

The multimodal claim is measured against two baselines, which keep separate two questions
that are easily blurred.

| Rung | Model | Question |
|---|---|---|
| (a) | stronger single modality alone | do two modalities beat one? |
| (b) | equal-weight average of the calibrated probability vectors | does a *learned* fusion beat a naive one? |
| (c) | the learned meta-learner | — the proposal |

Rung (a) is the comparator the hypothesis refers to: the fused model must beat the best
either modality manages alone. Rung (b) is deliberately trivial — it gives each modality
the same say on every class rather than learning how far to trust each — and isolates
whether the extra machinery of a learned fusion earns its place.

### 5.3 Two implementation choices

Both are departures from the specification as originally written and are stated explicitly.

**Input space: log-probabilities.** `z` holds log-probabilities rather than probabilities
(clipped at 10⁻¹²). Then `softmax(W log p + b)` is a weighted geometric mean — a product of
experts (Hinton, 2002) — and rung (b) sits *inside* the hypothesis space at the finite
point `W₀ = ½[I ; I]`, `b = 0`. With raw probabilities no such nesting exists, and the
branch outputs saturate near 0 and 1 (the tabular branch's mean maximum probability is
0.9999) where a linear map has almost no resolution. The raw-probability variant is still
fitted and reported as an ablation in §6.

**Penalty centre: the equal-weight point.** A standard L2 penalty shrinks `W → 0`, which
corresponds to ignoring both branches and predicting the marginal class prior — the wrong
prior for a fusion model. The penalty is instead centred on `W₀`, penalising `‖W − W₀‖²`
(and `‖b‖²`), so weak regularisation recovers the free stack and strong regularisation
recovers rung (b) *exactly*. The three rungs become a **continuum indexed by `C`**, and
rung (b) is the `C → 0` limit of rung (c) rather than a separate model. The notebook
asserts this numerically: at `C = 10⁻⁶` the stack's test probabilities match the
equal-weight average to within 10⁻³.

Because scikit-learn cannot express an offset penalty, the meta-learner is implemented in
PyTorch with LBFGS, mirroring the `TemperatureScaler` idiom already used in the stamp
branch.

### 5.4 Selecting the regularisation strength

`C` is selected by **stratified 5-fold cross-validation within the validation fold**; the
test fold is never consulted. The criterion is **log loss** rather than macro-F1: with
roughly a hundred AGN objects in validation, a per-fold macro-F1 is far too noisy to select
on, whereas log loss is a proper scoring rule that uses every object's full predicted
distribution. The regularisation path is shown in `03_C_selection`.

---

## 6. Results

The selected regularisation strength is `C = 100` (minimum within-validation CV log loss
0.0371, an interior optimum rather than a boundary — the path rises again at `C = 316`).
The fitted temperatures are 1.128 for the tabular branch and 1.667 for the image branch;
both branches were over-confident, the CNN markedly so, and scaling reduces its test ECE
from 0.041 to 0.010. The `C → 0` nesting check returns a maximum probability difference of
2.6 × 10⁻⁷ against the equal-weight baseline, confirming the re-centred penalty behaves as
designed.

### Test-set metrics

| Model | macro-F1 | balanced acc. | accuracy | MCC | weighted F1 | ROC-AUC | PR-AUC | log loss |
|---|---|---|---|---|---|---|---|---|
| (a) LightGBM (LC) alone | 0.9065 | **0.9315** | 0.9555 | 0.8450 | 0.9573 | 0.9892 | 0.9684 | 0.1371 |
| (a) EfficientNet-B0 alone | 0.7414 | 0.8059 | 0.9127 | 0.7038 | 0.9197 | 0.9539 | 0.7911 | 0.2419 |
| (b) Equal-weight average | 0.9091 | 0.8949 | 0.9730 | 0.8959 | 0.9727 | **0.9936** | 0.9705 | 0.0834 |
| **(c) Learned stack** | **0.9143** | 0.9030 | **0.9741** | **0.8999** | **0.9739** | 0.9935 | **0.9712** | **0.0816** |

Supporting rows (not part of the ladder):

| Variant | macro-F1 | balanced acc. | MCC | log loss |
|---|---|---|---|---|
| (c) Learned stack — raw probabilities | 0.9232 | 0.8840 | 0.9129 | 0.1230 |
| (c) Learned stack — Option B, deploy refit | 0.9485 | 0.9367 | 0.9450 | 0.0726 |

Paired bootstrap confidence intervals on the test macro-F1 differences (1 000 resamples,
`09_bootstrap_ci`, `bootstrap_ci.csv`):

| Comparison | Δ macro-F1 | 95% CI | Excludes zero |
|---|---|---|---|
| (c) learned stack vs (a) LightGBM alone | +0.0077 | [−0.0135, +0.0285] | no |
| (c) learned stack vs (b) equal-weight | +0.0052 | [−0.0154, +0.0248] | no |
| (b) equal-weight vs (a) LightGBM alone | +0.0026 | [−0.0249, +0.0265] | no |

Per-class test F1 (test support: SN 1 513, AGN 61, VS 201):

| Class | (a) LightGBM | image alone | (b) equal-weight | (c) learned stack | Δ (c) − (a) |
|---|---|---|---|---|---|
| SN | 0.9745 | 0.9534 | 0.9858 | **0.9865** | **+0.0120** |
| AGN | **0.9000** | 0.4671 | 0.8214 | 0.8348 | **−0.0652** |
| VS | 0.8451 | 0.8038 | 0.9201 | **0.9216** | **+0.0765** |

The learned weight matrix `W` (`07_weight_heatmap`, `fusion_weights.csv`); rows are output
classes, the first three columns the tabular block and the last three the image block. The
equal-weight prior sits at 0.5 on each diagonal:

| | tab SN | tab AGN | tab VS | img SN | img AGN | img VS | bias |
|---|---|---|---|---|---|---|---|
| **SN** | 0.627 | −0.041 | −0.126 | 0.584 | 0.073 | −0.132 | −0.006 |
| **AGN** | −0.136 | **0.671** | −0.007 | −0.093 | **0.308** | −0.036 | 0.048 |
| **VS** | 0.009 | −0.130 | 0.634 | 0.008 | 0.119 | **0.669** | −0.041 |

### Findings

- **The ladder is monotone, but no rung difference is statistically distinguishable.**
  Test macro-F1 rises 0.9065 → 0.9091 → 0.9143 across rungs (a), (b) and (c), so both
  questions receive the same qualitative answer: two modalities beat one, and a learned
  fusion beats a naive one. Every 95% bootstrap interval straddles zero. The defensible
  claim is that fusion does not harm and is probably mildly beneficial — not that a gain
  has been demonstrated on this test fold.

- **The probabilistic improvement is much clearer than the macro-F1 improvement.** Log loss
  falls 0.137 → 0.082 and MCC rises 0.845 → 0.900 — far larger relative movements than the
  +0.008 in macro-F1. Fusion's real contribution here is better-ordered, better-calibrated
  posteriors rather than a large number of newly-corrected argmax decisions. For a broker
  that thresholds on confidence to trigger follow-up, that is the more operationally
  relevant quantity.

- **The gain is concentrated in VS, and AGN pays for part of it.** VS F1 rises 0.845 → 0.922
  and SN 0.975 → 0.987, but AGN F1 *falls* 0.900 → 0.835 (recall 0.885 → 0.787). This is
  why balanced accuracy declines (0.932 → 0.903) even as macro-F1 rises. The mechanism is
  visible in the branch scores: the image branch is genuinely weak on AGN (F1 0.467, recall
  0.639), so admitting it dilutes the tabular branch's strong AGN performance, whereas on VS
  it carries real complementary signal.

- **The meta-learner discovered this per-class structure by itself.** The image diagonal is
  0.669 for VS — *higher* than the tabular 0.634, so the stack trusts the stamp more than
  the light curve for variable stars — but only 0.308 for AGN against a tabular 0.671. This
  is exactly the interpretability the low-dimensional late-fusion design was chosen to
  deliver: the weights read directly as per-class trust and they track where each modality
  is actually competent.

- **Learning the weights measurably limits the damage, which is the clearest evidence that
  rung (c) earns its place.** `08_perclass_delta` shows the naive average costs −0.079 AGN
  F1 against the tabular branch, while the learned stack costs −0.065 — it recovers about a
  fifth of the loss, precisely by down-weighting the image branch on the class where that
  branch is unreliable. On SN and VS the two rungs are indistinguishable (+0.011 vs +0.012,
  +0.075 vs +0.076). So the entire advantage of learning the fusion, on this dataset, is
  concentrated in damage limitation on the rare class — not in extra gain on the common
  ones. The learned stack still does not fully escape the AGN penalty on 61 test objects.

- **Fusion arbitrates between the branches rather than collapsing onto the stronger one.**
  On the 200 test objects where the branches disagree, the stack is correct 90.4% of the
  time when only the tabular branch is right (136 objects) and still 76.7% of the time when
  only the image branch is right (60 objects). A stack that had degenerated into copying the
  tabular vector would score near zero on the second figure. This is the behavioural
  counterpart to the weight matrix above, and it is only possible because §3 supplied a
  non-degenerate fitting fold.

- **The validation fold chose the blend weight almost optimally.** `06_blend_sensitivity`
  places the learned effective weight at *w* = 0.54, identical to the validation optimum,
  against a test optimum of 0.62. The validation fold pointed marginally too far toward the
  image branch, and the macro-F1 cost of that miss is small — a reassuring result for a
  fusion stage fitted on a fold that §3 had to rebuild.

- **The apparent image-branch temporal drift was an artefact of the contamination.** On the
  honest train-only models the image branch moves 0.7424 → 0.7414 from validation to test, a
  drift of −0.001; it is the *most* stable component in the system. The tabular branch drifts
  further (−0.028). Read from the published card figures instead, the image branch appears to
  collapse by 11.5 points — so any claim that the temporal split differentially punishes the
  image modality would have had the sign of the effect backwards. `11_drift` shows both
  readings side by side.

- **The raw-probability ablation scores higher on macro-F1 but is a worse probabilistic
  model.** Fitting on `z = [p_tab ; p_img]` directly gives 0.9232 macro-F1 against 0.9143 for
  log-probabilities, but log loss degrades from 0.082 to 0.123 and balanced accuracy from
  0.903 to 0.884. Given that the bootstrap intervals above are roughly ±0.02 wide, the
  macro-F1 difference is not meaningful, while the log-loss gap is substantial and consistent
  with the saturation argument of §5.3. The log-probability parameterisation is retained, and
  this is recorded as a case where the headline metric and the proper scoring rule disagree.

- **The Option A / Option B gap is large and expected.** Applying the same fitted fusion to
  the deployed train+val branch models yields 0.9485 test macro-F1 against 0.9143 for the
  train-only models. Almost all of that difference is the base models, not the fusion: the
  deployed LightGBM alone scores 0.9457 against 0.9065 for its train-only counterpart, simply
  because it was trained on 20% more data. Against its own baseline, Option B's fusion gain
  is +0.0028, smaller than Option A's +0.0077. The two provenances should never be compared
  across rows: the honest fusion gain is the within-provenance difference.

- **Fusion cost is negligible.** The meta-learner is a 3 × 6 matrix and a 3-vector — 24
  parameters, one matrix multiply and a softmax per object. The real-time budget is set
  entirely by the two branches, so the fusion stage does not affect the latency conclusions
  of either branch document.

---

## 7. Persistence contract

`models/fusion/logreg_stack/` follows the same contract as the two branch directories, so
the alert system can load all three the same way.

| File | Contents |
|---|---|
| `logreg_stack/meta_learner.pt` | torch state dict — `W` (3 × 6), `b` (3), and the penalty centre `W0` |
| `logreg_stack/fusion_card.json` | full provenance record; fields below |
| `logreg_stack/best_params.json` | chosen `C`, penalty centre, input space, CV protocol |
| `_probs_cache.npz` | train-only branch probabilities and logits, one level up in `models/fusion/` — regenerable, and skips both refits on re-run |
| `_probs_cache_key.json` | cache validity key (hash of both parameter files plus the split) and the fitted rounds / epochs: 243 boosting rounds, 15 epochs |

As with the two branch directories, `models/` is git-ignored, so these artefacts are
reproduced by running the notebook rather than pulled from the repository. A cold run
retrains the CNN (GPU minutes); a warm run completes in under a minute.

`fusion_card.json` records `fusion_type`, `input_space`, the two `branches` (family,
directory, fitted temperature, parameter hash), `base_model_provenance`, `input_columns`
(the column order of `z`), `W` and `b`, the `penalty` and `C_selection` protocol,
`class_names`, `dataset`, `taxonomy`, `split_hash` (both branch strings plus the
authoritative `oid_set_sha`), `train_date`, `val_metrics`, `test_metrics`, the two
`baselines`, the `blend` diagnostics, the full `leakage_audit` table, and
`package_versions`.

The column order of `z` is recorded explicitly as
`[log_p_tab_SN, log_p_tab_AGN, log_p_tab_VS, log_p_img_SN, log_p_img_AGN, log_p_img_VS]`,
because a silently transposed or reordered `z` at inference time would produce plausible
but wrong probabilities.

---

## 8. Risks and limitations

**The meta-learner is fitted on the validation fold without out-of-fold predictions.**
This follows the specification, and §3 restored a genuinely held-out fold, but the
statistically cleaner alternative — rolling-origin out-of-fold predictions over the
training fold — was not run because it requires retraining the CNN once per fold. With only
24 free parameters fitted on 1 773 objects, and with `C` chosen by cross-validation, the
residual optimism should be small, but it is not zero.

**Early stopping still touches validation.** The tabular boosting-round count and the image
stopping epoch are chosen on the validation fold even in the train-only refits. This is one
scalar per branch, and it matches how the light-curve branch computed its own clean
validation metrics, but the validation probabilities are not entirely free of it.

**Branch winners were selected on test.** As set out in §2, both branches chose their
carried model by test macro-F1 rather than validation macro-F1, and for the image branch
the two rules disagree. The branch test figures — and therefore rung (a) — carry the
optimism of a maximum-of-three (image) or maximum-of-four (tabular) selection.

**The hyper-parameter searches behind both branches were truncated.** The stamp studies
completed 3 of 20 and 5 of 20 trials. The carried image model is not a well-searched
configuration, which caps what the image branch contributes to fusion.

**A single temporal split.** There is one train/val/test cut, not repeated splits or a
rolling-origin evaluation, so every figure in §6 is a single realisation. The split is
also unstratified by construction, so class balance drifts between folds and val→test
comparisons conflate genuine drift with composition change.

**AGN is small.** With 581 AGN objects overall and roughly 60 in the test fold, per-class
AGN metrics are noisy, and macro-F1 — which weights AGN equally with SN — inherits that
noise. This is why §6 reports paired bootstrap intervals rather than point differences
alone.

**The Option A / Option B gap.** The headline result uses train-only base models while the
branch documents report train+val models. The two are not interchangeable, and the gap is
reported in §6 rather than assumed negligible.

**A TorchScript trap.** `models/stamp/effnet_b0/model_scripted.pt` does not upsample in
its graph: feeding it the native (N, 3, 63, 63) tensor returns near-chance output *with no
error*. The caller must bilinearly interpolate to 160 px first. The notebook asserts a
macro-F1 floor after the first inference pass so this cannot silently recur.

---

## 9. Figures index

| Figure | Contents |
|---|---|
| `01_leakage_audit` | Deployed (train+val-refit) models scored on train / val / test, per branch |
| `02_reliability` | Reliability diagrams, both branches, both folds, before and after temperature scaling |
| `03_C_selection` | Within-validation CV log loss and macro-F1 against `C`, with the selected value marked |
| `04_ladder_bar` | The three-rung ladder, validation and test paired |
| `05_confusion_grid` | Row-normalised confusion matrices for the three rungs |
| `06_blend_sensitivity` | Test macro-F1 across the one-parameter blend family, with rung (b), the val-optimal, learned-`W` effective and test-optimal weights marked |
| `07_weight_heatmap` | The learned `W` as a 3 × 6 heatmap, tabular and image blocks separated |
| `08_perclass_delta` | Per-class F1 change relative to the stronger single modality |
| `09_bootstrap_ci` | Paired bootstrap confidence intervals on the test macro-F1 differences between rungs |
| `10_disagreement` | Branch-disagreement breakdown, and whether the stack picks the right branch |
| `11_drift` | val → test drift per branch, contaminated against honest |

Accompanying tables: `test_metrics.csv`, `verdict.csv`, `fusion_weights.csv`,
`blend_sensitivity.csv`, `bootstrap_ci.csv`, `calibration.csv`, `leakage_audit.csv`.

---

## References

Carrasco-Davis, R., Reyes, E., Valenzuela, C., *et al.* (2021) 'Alert classification for the
ALeRCE broker system: the real-time stamp classifier', *The Astronomical Journal*, 162(6),
p. 231.

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017) 'On calibration of modern neural
networks', *Proceedings of the 34th International Conference on Machine Learning*, pp.
1321–1330.

Hinton, G. E. (2002) 'Training products of experts by minimizing contrastive divergence',
*Neural Computation*, 14(8), pp. 1771–1800.

Ke, G., Meng, Q., Finley, T., *et al.* (2017) 'LightGBM: a highly efficient gradient
boosting decision tree', *Advances in Neural Information Processing Systems*, 30, pp.
3146–3154.

Rehemtulla, N., Miller, A. A., Jegou Du Laz, T., *et al.* (2024) 'The Zwicky Transient
Facility Bright Transient Survey. III. BTSbot: automated identification and follow-up of
bright transients with deep learning', *The Astrophysical Journal*, 972(1), p. 7.

Sánchez-Sáez, P., Reyes, I., Valenzuela, C., *et al.* (2021) 'Alert classification for the
ALeRCE broker system: the light curve classifier', *The Astronomical Journal*, 161(3),
p. 141.

Tan, M. and Le, Q. (2019) 'EfficientNet: rethinking model scaling for convolutional neural
networks', *Proceedings of the 36th International Conference on Machine Learning*, pp.
6105–6114.

Ting, K. M. and Witten, I. H. (1999) 'Issues in stacked generalization', *Journal of
Artificial Intelligence Research*, 10, pp. 271–289.

Wolpert, D. H. (1992) 'Stacked generalization', *Neural Networks*, 5(2), pp. 241–259.
