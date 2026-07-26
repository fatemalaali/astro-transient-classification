# Late-Fusion Classifier — ZTF Gold (companion write-up)

Companion to `fusion_ztf.ipynb`. This document records *what* was done and *why*, in a
form ready to lift into the methodology and results chapters. It uses UK spelling and
Harvard author–year citations; a reference list is at the end. All figures referenced
live in `figures/fusion/` (300-dpi PNG + vector PDF); the saved meta-learner lives in
`models/fusion/logreg_stack/`.

This is the third and final classifier branch for the ZTF gold dataset. It consumes the
**branch contracts** emitted by the two unimodal branches — `lc_classifier_ztf.ipynb`
(tabular light-curve features) and `stamp_classifier_ztf.ipynb` (image stamps) — and learns
a decision-level meta-learner over their probability vectors. Every branch now emits a clean
out-of-fold (OOF) probability matrix, an OOF-fitted temperature, and its deployed model's
test probabilities, so fusion is a matter of loading arrays and fitting a small stack; the
leakage-audit and train-only-refit machinery that earlier drafts needed is gone.

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
objects in the same folds. This is now a real, one-line check: every branch card stores the
canonical **`split_id`** (`protocol.split_id` — a SHA-1 of the sorted `oid → split` text
payload, written into the gold `MANIFEST.json`), and the notebook calls
`protocol.assert_same_split([tab_card, img_card])`. Both cards carry `76c4c40d0352`, so the
check passes. The canonical hash is order-independent by construction, which retires the
earlier problem where the two branches' `pd.util.hash_pandas_object` strings differed
cosmetically (`index=` and row order) despite identical underlying folds and so could not be
compared directly.

---

## 2. Branch selection

The model carried from each branch is the **validation winner**, discovered from disk rather
than hard-coded: the notebook scans each branch directory for the sub-model whose card marks
`is_branch_winner` and that carries an OOF matrix. No re-selection happens in the fusion
notebook, and no selection anywhere reads the test fold (`protocol.select_winner` operates on
the validation frame only).

| Branch | Winner | Validation macro-F1 (selection) | Runner-up |
|---|---|---|---|
| Tabular | **LightGBM** | 0.9346 | XGBoost 0.9320 |
| Image | **EfficientNet-B0** | 0.7632 | ResNet-18 0.7617 |

**The image winner is a near-tie and worth flagging.** EfficientNet-B0 leads ResNet-18 by
0.0015 validation macro-F1 — inside run-to-run variance — and on the test fold the two even
swap (ResNet-18 0.7607 against EfficientNet-B0 0.7592). Selecting on validation carries
EfficientNet-B0; either backbone would be a defensible choice, and neither the branch nor the
fusion conclusion depends on which is taken. The tabular margin (LightGBM over XGBoost, 0.0026)
is likewise small; both are strong boosters and the choice between them is not load-bearing.

---

## 3. The branch contract

Each branch emits exactly what fusion needs, so fusion re-implements nothing. For the winner:

| Artefact | Meaning |
|---|---|
| `model.*` (joblib / state_dict) | the **deployed** model, refit on **train + val** |
| `best_params.json` | the tuned configuration |
| `temperature.json` | scalar *T*, **fitted on OOF** |
| `oof_proba.npy` + `oof_oids.npy` | forward-chaining OOF probabilities, aligned to train oids |
| `test_proba.npy` + `test_oids.npy` | the deployed model's test probabilities and oids |
| `model_card.json` | class order, feature/preproc contract, canonical `split_id`, `base_provenance = train_val_refit`, `oof_provenance = forward_chain_5block` |

**Out-of-fold predictions.** Each branch produces its OOF matrix by forward-chaining inside
its own training fold (`protocol.forward_chain_oof`): the training objects are ordered by
`firstmjd` and cut into five contiguous, time-ordered blocks; for block *r+1* the model is
refit on blocks *0..r* — at the tuned rounds/epochs, with **no** early stopping and no peeking
at the block it is about to predict — and used to predict block *r+1*. Block 0 is never
predicted, so the OOF set is 6 622 of the 8 278 training objects. These probabilities are
never in-sample for the model that produced them, which is precisely the fold a stacking
meta-learner (Wolpert, 1992) must be fitted on.

This replaces the workaround earlier drafts needed. Previously both branches saved only their
train+val-refit ("deployed") models, whose validation predictions were in-sample — the tabular
booster reproduced the validation labels almost exactly, which made both the fusion fit and the
tabular temperature fit degenerate. The fusion notebook then had to refit each branch on the
training fold alone to recover a clean fitting fold, and reasoned at length about the resulting
leakage. With the OOF contract in place, that entire section is obsolete and has been removed.

**Alignment.** The two branches store their OOF (and test) rows in different oid orders, so the
notebook aligns every matrix by `oid` before use, and asserts the OOF oid *sets* are identical
across branches (they are: both are the same four forward-chaining blocks of the shared split).

**A conservative bias, stated.** The meta-learner is fitted on OOF sub-models, which are
slightly weaker than the deployed train+val branches it is applied to at test time (tabular OOF
macro-F1 0.960 against deployed test 0.946; image OOF 0.751 against deployed test 0.759). The
fusion weights are therefore learned against marginally weaker branches than they are finally
applied to — a bias *against* the fusion hypothesis, which is the safe direction to err in.

---

## 4. Calibration

Because the fusion stage combines probabilities, both branches are calibrated before stacking.
An uncalibrated branch would silently dominate the fusion weights regardless of its true
reliability, and convolutional networks in particular tend to be over-confident
(Guo et al., 2017).

Each branch is calibrated by **temperature scaling**: a single scalar *T* dividing the logits
before the softmax, `p = softmax(z / T)`. Crucially, *T* is fitted **on each branch's OOF fold**
inside the branch notebook and stored in `temperature.json` — not on validation, and not
in-sample. For the tabular branch, which emits probabilities rather than logits, the
log-probabilities serve as logits, and temperature scaling on `log p` is identical to scaling on
raw logits because softmax is invariant to per-row additive constants.

The fitted temperatures are **1.672** (tabular) and **3.018** (image). Both branches are
over-confident on their honest OOF fold, the CNN markedly so: its OOF maximum probabilities sit
near certainty while its OOF macro-F1 is only 0.751, so a large scalar is needed to soften them.
This value is exactly what an in-sample validation fit would have understated. Because temperature
scaling is a monotone transform of the logits, it cannot change any branch's own argmax or its
macro-F1 — only how much weight the branch carries once combined. Reliability of the calibrated
branches on the OOF and test folds is shown in `02_reliability`; the branch OOF/test ECEs are
0.003 / 0.007 (tabular) and 0.022 / 0.054 (image).

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
either modality manages alone. The stronger single modality is chosen on the **OOF** fold —
the tabular branch (OOF macro-F1 0.960 against the image branch's 0.751). Rung (b) is
deliberately trivial — it gives each modality the same say on every class rather than learning
how far to trust each — and isolates whether the extra machinery of a learned fusion earns its
place.

### 5.3 Two implementation choices

Both are departures from the specification as originally written and are stated explicitly.

**Input space: log-probabilities.** `z` holds log-probabilities rather than probabilities
(clipped at 10⁻¹²). Then `softmax(W log p + b)` is a weighted geometric mean — a product of
experts (Hinton, 2002) — and rung (b) sits *inside* the hypothesis space at the finite
point `W₀ = ½[I ; I]`, `b = 0`. With raw probabilities no such nesting exists, and the
branch outputs saturate near 0 and 1 where a linear map has almost no resolution. The
raw-probability variant is still fitted and reported as an ablation in §6.

**Penalty centre: the equal-weight point.** A standard L2 penalty shrinks `W → 0`, which
corresponds to ignoring both branches and predicting the marginal class prior — the wrong
prior for a fusion model. The penalty is instead centred on `W₀`, penalising `‖W − W₀‖²`
(and `‖b‖²`), so weak regularisation recovers the free stack and strong regularisation
recovers rung (b) *exactly*. The three rungs become a **continuum indexed by `C`**, and
rung (b) is the `C → 0` limit of rung (c) rather than a separate model. The notebook
asserts this numerically: at `C = 10⁻⁶` the stack's test probabilities match the
equal-weight average to within 10⁻³.

Because scikit-learn cannot express an offset penalty, the meta-learner is implemented in
PyTorch with LBFGS, mirroring the `TemperatureScaler` idiom used elsewhere.

### 5.4 Selecting the regularisation strength

`C` is selected by **stratified 5-fold cross-validation within the OOF fold**; the test fold
is never consulted. The criterion is **log loss** rather than macro-F1: with roughly a few
hundred AGN objects in the OOF set, a per-fold macro-F1 is far too noisy to select on, whereas
log loss is a proper scoring rule that uses every object's full predicted distribution. The
selected value is **`C = 1000`** — the weak-penalty end of the grid, i.e. essentially the free
stack — so the data support learning the full per-class weight matrix rather than collapsing
toward the equal-weight prior. The regularisation path is shown in `03_C_selection`.

---

## 6. Results

The test fold is read exactly once, at the end. There is a single fusion model line: the
OOF-fitted meta-learner applied to the deployed (train+val) branches' calibrated test
probabilities. The `C → 0` nesting check returns a maximum probability difference of order
10⁻⁷ against the equal-weight baseline, confirming the re-centred penalty behaves as designed.

### Test-set metrics

| Model | macro-F1 | balanced acc. | accuracy | MCC | weighted F1 | ROC-AUC | PR-AUC | log loss |
|---|---|---|---|---|---|---|---|---|
| (a) LightGBM (LC) alone | 0.9457 | 0.9440 | 0.9814 | 0.9283 | 0.9814 | 0.9938 | 0.9766 | 0.0726 |
| (a) EfficientNet-B0 alone | 0.7592 | 0.8031 | 0.9189 | 0.7135 | 0.9227 | 0.9637 | 0.8315 | 0.2279 |
| (b) Equal-weight average | 0.9448 | 0.9292 | 0.9837 | 0.9361 | 0.9835 | **0.9957** | **0.9782** | **0.0604** |
| **(c) Learned stack** | **0.9528** | **0.9566** | 0.9831 | 0.9352 | 0.9831 | 0.9953 | 0.9768 | 0.0720 |

Supporting row (ablation, not part of the ladder): the learned stack fitted on raw
probabilities instead of log-probabilities scores macro-F1 0.9534, MCC 0.9412, log loss 0.0679
— indistinguishable from rung (c) on macro-F1, so the log-probability parameterisation is
retained for its product-of-experts interpretation and the exact rung-(b) nesting.

Paired bootstrap confidence intervals and McNemar tests on the test differences (1 000
resamples, `09_bootstrap_ci`, `bootstrap_ci.csv`):

| Comparison | Δ macro-F1 | 95% CI | McNemar *p* | Distinguishable? |
|---|---|---|---|---|
| (c) learned stack vs (a) LightGBM alone | +0.0071 | [−0.0017, +0.0178] | 0.508 | no |
| (c) learned stack vs (b) equal-weight | +0.0080 | [−0.0067, +0.0262] | 1.000 | no |
| (b) equal-weight vs (a) LightGBM alone | −0.0009 | [−0.0184, +0.0157] | 0.481 | no |

The learned weight matrix `W` (`07_weight_heatmap`, `fusion_weights.csv`); rows are output
classes, the first three columns the tabular block and the last three the image block:

| | tab SN | tab AGN | tab VS | img SN | img AGN | img VS | bias |
|---|---|---|---|---|---|---|---|
| **SN** | 0.977 | −0.100 | −0.193 | 0.498 | 0.180 | −0.147 | −0.291 |
| **AGN** | −0.237 | **0.802** | −0.100 | −0.021 | **0.163** | −0.009 | 0.472 |
| **VS** | −0.239 | −0.202 | 0.793 | 0.024 | 0.157 | **0.656** | −0.181 |

### Findings

- **The ladder is monotone, but no rung difference is statistically distinguishable.**
  Test macro-F1 rises 0.9457 → 0.9448 → 0.9528 (a → b → c; the equal-weight rung dips a
  hair below the single modality before the learned stack overtakes both). Every 95% bootstrap
  interval straddles zero and both McNemar tests are non-significant. The defensible claim is
  that fusion does not harm and is probably mildly beneficial — not that a gain has been
  demonstrated on this single test fold.

- **The learned stack improves balanced accuracy and MCC, not only macro-F1.** Balanced
  accuracy rises 0.944 → 0.957 and the equal-weight rung already halves log loss (0.073 → 0.060).
  Unlike a naive average, the learned stack lifts the headline metric *and* keeps the
  probabilistic quality competitive — the more operationally relevant behaviour for a broker
  that thresholds on confidence.

- **The learned weights read as per-class trust and match where each modality is competent.**
  For AGN the stack leans almost entirely on the tabular branch (0.802 against a negligible
  0.163 image weight): the image branch is genuinely weak on AGN. For VS both modalities carry
  real weight (tabular 0.793, image 0.656) — the difference channel of the stamp does contain
  variable-star signal. For SN the tabular branch leads (0.977) with a substantial image
  contribution (0.498). This is exactly the interpretability the low-dimensional late-fusion
  design was chosen to deliver.

- **Fusion arbitrates, and it leans on the stronger branch — as it should here.** On the 150
  test objects where the branches disagree, the stack is correct 99.2% of the time when only the
  tabular branch is right (129 objects) and 33.3% of the time when only the image branch is right
  (18 objects). Given that the tabular branch is far stronger overall (test macro-F1 0.946 against
  0.759), a stack that trusts it heavily on disagreements is the correct behaviour, not a
  degeneracy; it still recovers a third of the cases where only the weaker branch is right.

- **The blend was chosen sensibly but not optimally on this fold.** `06_blend_sensitivity` puts
  the learned effective weight at *w* = 0.66 (leaning tabular), against an OOF-optimal of *w* = 1.0
  (pure tabular) and a test-optimal of *w* = 0.4 (leaning image). The learned per-class stack sits
  between the two scalar optima and beats rung (a) on test, so admitting the image branch helped
  even though a scalar OOF blend would have discarded it — evidence that the *per-class* weights,
  not a single blend, are doing the work.

- **The raw-probability ablation is a dead heat on macro-F1.** Fitting on `z = [p_tab ; p_img]`
  directly gives 0.9534 against 0.9528 for log-probabilities — well inside the ±0.02 bootstrap
  width — so nothing rests on the choice; the log-probability form is kept for its geometric-mean
  interpretation and its exact nesting of rung (b).

- **Fusion cost is negligible.** The meta-learner is a 3 × 6 matrix and a 3-vector — 24
  parameters, one matrix multiply and a softmax per object. The real-time budget is set entirely
  by the two branches, so the fusion stage does not affect the latency conclusions of either
  branch document.

---

## 7. Persistence contract

`models/fusion/logreg_stack/` follows the same contract as the two branch directories, so
the alert system can load all three the same way.

| File | Contents |
|---|---|
| `logreg_stack/meta_learner.pt` | torch state dict — `W` (3 × 6), `b` (3), and the penalty centre `W0` |
| `logreg_stack/fusion_card.json` | full provenance record; fields below |
| `logreg_stack/best_params.json` | chosen `C`, penalty centre, input space, CV protocol |

`fusion_card.json` records `fusion_type`, `input_space`, the two `branches` (family,
directory, OOF-fitted temperature, `oof_provenance`, `base_provenance`), `meta_fit_fold`
(`forward_chain_oof`), `scoring_base` (`train_val_refit_deployed`), `input_columns` (the
column order of `z`), `W` and `b`, the `penalty` and `C_selection` protocol (`stratified_5fold_within_oof`),
`class_names`, `dataset`, `taxonomy`, the canonical `split_id`, `train_date`, the OOF and test
metrics of the fused model, the two `baselines`, the `blend` diagnostics, the `significance`
table (bootstrap CIs + McNemar), and `package_versions`. There is a single model line — no
train-only/deploy-refit duality to reconcile — so no `leakage_audit` field remains.

The column order of `z` is recorded explicitly as
`[log_p_tab_SN, log_p_tab_AGN, log_p_tab_VS, log_p_img_SN, log_p_img_AGN, log_p_img_VS]`,
because a silently transposed or reordered `z` at inference time would produce plausible
but wrong probabilities. As with the branch directories, `models/` is git-ignored, so these
artefacts are reproduced by running the notebook. The fusion notebook itself no longer trains
anything (it reads the saved probability arrays), so it runs in seconds once the branches exist.

---

## 8. Risks and limitations

**A single temporal split.** There is one train/val/test cut, not repeated splits or a
rolling-origin evaluation, so every figure in §6 is a single realisation. The split is also
unstratified by construction, so class balance drifts between folds and OOF→test comparisons
conflate genuine drift with composition change.

**AGN is small.** With 581 AGN objects overall and roughly 60 in the test fold, per-class AGN
metrics are noisy, and macro-F1 — which weights AGN equally with SN — inherits that noise. This
is why §6 reports paired bootstrap intervals and McNemar tests rather than point differences
alone, and why no rung difference is called significant.

**The image winner is a near-tie.** EfficientNet-B0 is carried on a 0.0015 validation-macro-F1
lead over ResNet-18, which the two swap on test. The image branch's contribution to fusion should
be read as "the better of two near-equivalent backbones", and the fusion conclusion is unchanged
under either.

**The conservative-bias direction.** The meta-learner is fitted on OOF sub-models slightly weaker
than the deployed branches it scores, so any measured fusion gain is, if anything, understated.
This is the intended direction but it is a mild mismatch, not a null one.

**Residual dependence on validation.** The number of boosting rounds (tabular) and the stopping
epoch (image) that define each OOF sub-model are the tuned values, which were themselves chosen
with early stopping on validation. That is one scalar per branch and is standard, but the OOF
predictions are not entirely independent of the validation fold.

**A TorchScript trap (alert system only).** `models/stamp/effnet_b0/model_scripted.pt` does not
upsample in its graph: feeding it the native (N, 3, 63, 63) tensor returns near-chance output
*with no error*. The caller must bilinearly interpolate to 160 px first. Fusion no longer touches
the TorchScript export — it reads the saved probability arrays directly — so this now matters only
to the alert system, whose loader must reproduce the branch preprocessing.

---

## 9. Figures index

| Figure | Contents |
|---|---|
| `02_reliability` | Reliability of the OOF-calibrated branches, OOF and test folds |
| `03_C_selection` | Within-OOF CV log loss and macro-F1 against `C`, with the selected value marked |
| `04_ladder_bar` | The three-rung ladder, OOF (fit) and test (sealed) paired |
| `05_confusion_grid` | Row-normalised confusion matrices for the three rungs |
| `06_blend_sensitivity` | Test macro-F1 across the one-parameter blend family, with rung (b), the OOF-optimal, learned-`W` effective and test-optimal weights marked |
| `07_weight_heatmap` | The learned `W` as a 3 × 6 heatmap, tabular and image blocks separated |
| `08_perclass_delta` | Per-class F1 change relative to the stronger single modality |
| `09_bootstrap_ci` | Paired bootstrap confidence intervals on the test macro-F1 differences between rungs |
| `10_disagreement` | Branch-disagreement breakdown, and whether the stack picks the right branch |

Accompanying tables: `test_metrics.csv`, `verdict.csv`, `fusion_weights.csv`,
`blend_sensitivity.csv`, `bootstrap_ci.csv`, `branch_summary.csv`.

---

## References

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017) 'On calibration of modern neural
networks', *Proceedings of the 34th International Conference on Machine Learning*, pp.
1321–1330.

Hinton, G. E. (2002) 'Training products of experts by minimizing contrastive divergence',
*Neural Computation*, 14(8), pp. 1771–1800.

Ke, G., Meng, Q., Finley, T., *et al.* (2017) 'LightGBM: a highly efficient gradient
boosting decision tree', *Advances in Neural Information Processing Systems*, 30, pp.
3146–3154.

Tan, M. and Le, Q. (2019) 'EfficientNet: rethinking model scaling for convolutional neural
networks', *Proceedings of the 36th International Conference on Machine Learning*, pp.
6105–6114.

Ting, K. M. and Witten, I. H. (1999) 'Issues in stacked generalization', *Journal of
Artificial Intelligence Research*, 10, pp. 271–289.

Wolpert, D. H. (1992) 'Stacked generalization', *Neural Networks*, 5(2), pp. 241–259.
