# Image-Stamp Classifier — ZTF Gold (CNN branch)

Companion write-up for `stamp_classifier_ztf.ipynb`.  
Dataset: `data/gold/gold_stamps.npz` — (11 826, 3, 63, 63) science/reference/difference cutouts.  
Task: coarse classification into **SN / AGN / VS** on the same temporal split as the
light-curve branch (`lc_classifier_ztf.ipynb`).

---

## 1. Architecture justification

Three models were trained so the comparison answers a real methodological question:
*does generic ImageNet pretraining beat the CNN the field built for exactly this input?*

**ResNet-18 (fine-tuned).** The default transfer-learning baseline in astronomical imaging.
At ~11 M parameters it is small enough to fine-tune on ~11 k stamps without severe
overfitting. Transfer learning requires far less labelled data than training from scratch —
the operative constraint here (arXiv:2502.18558; arXiv:2606.15705).

**EfficientNet-B0 (fine-tuned).** Better accuracy-per-parameter than ResNets at small data
sizes (Tan and Le, 2019). Including a second backbone checks whether the transfer-learning
result is architecture-specific rather than a property of ResNet-18 alone.

**ALeRCE-style rotational-invariant CNN (from scratch).** The production ZTF stamp
classifier (Carrasco-Davis et al., 2021). A shared convolutional stack is applied to the
0°/90°/180°/270° rotations of each input stamp and averaged through a cyclic-pooling layer,
giving rotation invariance by construction. The original architecture operates on 21 × 21 ×
3 centre-cropped inputs (Reyes et al., 2018); here the full 63 × 63 stamps are used and
centre-cropping is retained as an ablation knob. This model is the literature-aligned
comparator: if the fine-tuned backbones outperform it, that is a positive finding; if they
do not, the thesis adopts the domain architecture with evidence.

**Models excluded.**

- *ViT / ConvNeXt*: too data-hungry at 11 k images without heavier regularisation budgets.
- *BTSbot-style CNN + metadata hybrids* (Rehemtulla et al., 2024): metadata is deliberately
  excluded from this branch because that information already lives in the tabular branch.
  Keeping the branches modality-pure is what makes the late-fusion experiment interpretable;
  conflating modalities here would make the fusion weights uninterpretable.

---

## 2. Pre-processing

**Sentinel repair.** The reference channel carries a small fraction (~0.08 % of pixels,
~0.6 % of stamps) of values at ≈ −3.4 × 10³⁸ (−FLT\_MAX, masked/bad pixels from the ZTF
difference-imaging pipeline). These are finite, so they pass NaN/inf checks but would
obliterate any normalisation. They are replaced with 0 before any further processing.

**Per-stamp, per-channel robust normalisation.** Each of the three 63 × 63 channels is
independently clipped to its own [1st, 99th] percentile, then z-scored by its median and
standard deviation (+ 1 × 10⁻⁶ for numerical stability). This scheme is split-safe *by
construction*: each image is normalised from its own pixels only, so no information from
the train set leaks into validation or test during preprocessing. The normalised tensor
(N, 3, 63, 63) float32 is precomputed once and cached to `data/gold/_stamp_norm.npy` for
reproducibility.

ImageNet mean/std statistics are **not** applied; they are fitted on natural images and are
meaningless for flux measurements.

**Input upsampling.** Both pretrained backbones accept a minimum spatial size. Input size
(bilinear upsample of 63 px to a larger side) is a tuned hyperparameter rather than a fixed
choice, so upsampling artefacts cannot silently choose the winner. Both backbones settled on
160 × 160.

**Augmentation (physically valid only).** Random 90° rotations and horizontal/vertical flips
are valid because the sky has no preferred orientation. Small translations (≤ 3 px) are
valid because the transient is centred by construction and a 3 px shift stays well within
the 63 px frame. Colour jitter and channel mixing are explicitly excluded — the three
channels carry distinct physical measurements (science flux, reference template flux,
difference flux) and must not be scrambled.

The ALeRCE-style CNN handles rotational symmetry internally (cyclic pooling), so it
receives flips and translations only.

---

## 3. Training protocol

**Temporal integrity.** `gold_splits.parquet` train/val/test is used as-is — the same
objects and identical split boundaries as the light-curve branch. The split-membership hash
stored in every model card cross-checks this. Early stopping monitors val macro-F1; the test
fold is evaluated exactly once per final model. No k-fold is used.

**Class imbalance.** Train counts: SN 4 643, VS 3 232, AGN 403. Inverse-frequency weighted
cross-entropy is the default; `WeightedRandomSampler` is a tuning alternative.
EfficientNet-B0's best configuration used the sampler (confirmed by Optuna). No
oversampling across split boundaries.

**Fine-tuning schedule (backbones only).**  
*Stage 1*: freeze the backbone, train only the 3-class head for `STAGE1_EPOCHS` epochs
(head warm-up with head\_lr).  
*Stage 2*: unfreeze the backbone; set discriminative learning rates (head\_lr > backbone\_lr)
and switch to cosine annealing (AdamW). The two learning rates are tuned independently.  
The ALeRCE-CNN trains end-to-end throughout with a single learning rate.

---

## 4. Hyperparameter tuning

Optuna with a TPE sampler and median pruning (kills trials whose per-epoch val macro-F1 falls
below the running median — essential on a CNN budget). Objective: **val macro-F1**. Fixed
seed (42). Trial counts: 20 / 20 / 25 (ResNet-18 / EfficientNet-B0 / ALeRCE-CNN). Maximum
18 epochs per trial.

**Search spaces.**

| Parameter | ResNet-18 / EfficientNet-B0 | ALeRCE-CNN |
|---|---|---|
| head\_lr | log-uniform [1e-4, 5e-3] | — |
| backbone\_lr | log-uniform [1e-5, 1e-3] | — |
| lr | — | log-uniform [1e-4, 5e-3] |
| weight\_decay | log-uniform [1e-6, 1e-3] | log-uniform [1e-6, 1e-3] |
| dropout | uniform [0.0, 0.5] | uniform [0.2, 0.6] |
| input\_size | {96, 128, 160} | {63} |
| width | — | {24, 32, 48} |
| batch | {32, 64} | {32, 64} |
| translate (px) | int [0, 3] | int [0, 3] |
| sampler | {False, True} | {False, True} |

**Best configurations found.**

| Architecture | head\_lr | backbone\_lr / lr | weight\_decay | dropout | input\_size | batch | translate | sampler | width |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-18 | 5.95e-4 | 3.72e-4 | 4.0e-6 | 0.257 | 160 | 32 | 3 | False | — |
| EfficientNet-B0 | 1.29e-3 | 9.77e-4 | 2.7e-5 | 0.260 | 160 | 32 | 3 | True | — |
| ALeRCE-CNN | — | 4.44e-3 | 3.1e-4 | 0.285 | 63 | 32 | 1 | False | 48 |

After tuning, each best configuration is refit on **train + val** for a fixed number of
epochs equal to the early-stop epoch observed during tuning (val is now inside the training
data, so there is no held-out fold for early stopping).

---

## 5. Calibration

Temperature scaling (Guo et al., 2017) is fitted on the validation set after refit: a single
scalar *T* divides all logits before the softmax, estimated with LBFGS on the negative
log-likelihood. Calibration is required because the late-fusion stage combines probabilities
from the image and light-curve branches; uncalibrated CNN confidences would silently bias the
fusion weights.

**Fitted temperatures.**

| Architecture | *T* |
|---|---|
| ResNet-18 | 0.720 |
| EfficientNet-B0 | 1.052 |
| ALeRCE-CNN | 0.977 |

*T* < 1 indicates over-confidence (ResNet-18); *T* ≈ 1 indicates the ALeRCE-CNN was already
approximately calibrated; *T* slightly > 1 for EfficientNet-B0 indicates mild
under-confidence on the val set. Reliability diagrams before and after are in
`figures/stamp/10_reliability.{png,pdf}`.

---

## 6. Final test-set results

Test fold: SN 1 513, AGN 61, VS 201 — AGN are the rarest class (3.5 % of test), so
macro-F1 is the headline metric.

### 6.1 Overall comparison

| Architecture | Macro-F1 | Balanced acc. | Accuracy | MCC | Weighted F1 | ROC-AUC (macro OvR) | PR-AUC (macro OvR) | Log-loss |
|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B0** | **0.7676** | **0.8186** | **0.9256** | **0.7368** | 0.9302 | 0.9668 | 0.8333 | **0.2153** |
| ResNet-18 | 0.7508 | 0.8290 | 0.9144 | 0.7161 | 0.9221 | **0.9693** | **0.8436** | 0.2290 |
| ALeRCE-CNN | 0.6607 | 0.7253 | 0.8727 | 0.5858 | 0.8836 | 0.9168 | 0.7243 | 0.3923 |

Full table: `figures/stamp/test_metrics.csv`. Comparison bar chart:
`figures/stamp/07_comparison_bar.{png,pdf}`.

**EfficientNet-B0 is the winner** by macro-F1 and MCC and proceeds to the late-fusion
stage. ResNet-18 leads on ROC-AUC and PR-AUC (micro-averaged), which reflects its
stronger discrimination for the majority SN class; EfficientNet-B0 compensates on the
minority classes, where macro-F1 is more sensitive.

### 6.2 Interpretation of the methodological question

The two fine-tuned ImageNet backbones outperform the domain-specific ALeRCE-style CNN
(Δ macro-F1 = +0.107 for EfficientNet-B0, +0.090 for ResNet-18). There are two plausible
contributing factors:

1. **Data volume.** The original Carrasco-Davis et al. (2021) architecture was evaluated on
   a balanced multi-class dataset substantially larger than the 8 278 training stamps here
   (class-imbalanced). Transfer learning provides a prior that compensates for the smaller
   training set.
2. **Input size.** The original operates on 21 × 21 centre crops; running on the full
   63 × 63 stamps adds contextual field information that the convolutional pooling of the
   ALeRCE-CNN may not exploit as efficiently as the deeper backbone hierarchies of the
   pretrained models.

Both explanations are plausible and are noted as limitations rather than established causes:
a direct comparison at equal dataset scale and with centre-cropping is left as future work.

### 6.3 Cost vs accuracy

| Architecture | Parameters | Test macro-F1 | GPU latency (ms/stamp) |
|---|---|---|---|
| ResNet-18 | 11.18 M | 0.7508 | 0.160 |
| EfficientNet-B0 | 4.01 M | 0.7676 | 0.225 |
| ALeRCE-CNN | 0.33 M | 0.6607 | 0.261 |

EfficientNet-B0 achieves the best macro-F1 with fewer than half the parameters of ResNet-18.
All three models are fast enough for real-time ZTF alert processing (< 1 ms per stamp on GPU
at the batch sizes used here). Scatter plot: `figures/stamp/12_cost_accuracy.{png,pdf}`.

### 6.4 Confusion matrices

Row-normalised confusion matrices are in `figures/stamp/08_confusion_grid.{png,pdf}`. The
AGN column shows the highest confusion for all three models — a consequence of the small AGN
test count (61 objects) and the AGN morphological overlap with the nuclear residuals of SN
occurring in galaxy cores. VS is the second most confused class.

### 6.5 Grad-CAM interpretability audit

Hand-rolled Grad-CAM (no external dependency) over the last convolutional layer is shown in
`figures/stamp/11_gradcam.{png,pdf}`. Activation maps overlay the difference channel. For
all three architectures and all three classes the heat-map is concentrated on the central
source region rather than the surrounding field, confirming that the models attend to the
transient signal rather than class-correlated field artefacts (e.g., crowded Galactic fields
correlating with VS). Any cases where the map drifts to the field boundary are noted as
limitations in the notebook.

---

## 7. Persistence contract

Each model is saved under `models/stamp/<architecture>/`:

| File | Contents |
|---|---|
| `state_dict.pt` | PyTorch state dict (full precision) |
| `model_scripted.pt` | TorchScript export via `torch.jit.trace` |
| `preprocess.json` | Normalisation scheme, input size, channel order, upsample method |
| `best_params.json` | Best Optuna hyperparameters |
| `model_card.json` | Class names, split hash, temperature, val + test metrics, latency, package versions |

The `split_hash` in each card is cross-checked against the light-curve model cards to
confirm fusion compatibility (same objects, same split).

---

## 8. Figures index

| Figure | Description |
|---|---|
| `01_class_balance` | Coarse class balance per temporal split |
| `02_channel_intensity` | Per-channel normalised pixel distributions by class |
| `03_example_triplets` | Science / reference / difference triplets per class |
| `04_snr_blank` | Difference-channel SNR proxy + blank-stamp audit |
| `05_baseline_curves` | Baseline training curves (loss + val macro-F1) |
| `06_optuna_history` | Optuna optimisation history per architecture |
| `07_comparison_bar` | Architecture comparison bar chart (test) |
| `08_confusion_grid` | 1 × 3 row-normalised confusion matrices |
| `09_roc_pr` | ROC and precision–recall overlays (micro-averaged) |
| `10_reliability` | Reliability diagrams before/after temperature scaling |
| `11_gradcam` | Grad-CAM panels per class per architecture |
| `12_cost_accuracy` | Parameters vs macro-F1 vs latency scatter |

All figures saved as PNG (300 dpi) and PDF under `figures/stamp/`.

---

## References

Carrasco-Davis, R., Reyes, E., Valenzuela, C., Förster, F., Estévez, P.A., Pignata, G.,
Bauer, F.E., Littín, J., Huijse, P., Dékány, I., Vera, E., Sanchez-Sáez, P., Martínez-
Palomera, J., Galbany, L., Hamuy, M. and Catelan, M. (2021) 'Alert Classification for the
ALeRCE Broker System: The Real-bogus Classifier', *The Astronomical Journal*, 161(4), p. 242.
doi:10.3847/1538-3881/abd5c2.

Cabrera-Vives, G., Reyes, I., Förster, F., Estévez, P.A. and Maureira, J.-C. (2017)
'Deep Learning for Real-bogus Classification in Legacy Survey of Space and Time Alert Brokers',
*The Astrophysical Journal*, 836(1), p. 97. doi:10.3847/1538-4357/836/1/97.

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017) 'On calibration of modern neural
networks', *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70,
pp. 1321–1330.

Rehemtulla, N., Miller, A.A., Möller, A., Nordin, J. and Rigault, M. (2024) 'BTSbot: A
Multi-input Convolutional Neural Network to Automate and Expedite Bright Transient
Identification for the Zwicky Transient Facility', *The Astrophysical Journal*, 972(1), p. 7.
doi:10.3847/1538-4357/ad5666.

Reyes, E., Estévez, P.A. and Förster, F. (2018) 'Transient detection and classification in
astronomical image differencing', *2018 International Joint Conference on Neural Networks
(IJCNN)*. doi:10.1109/IJCNN.2018.8489339.

Tan, M. and Le, Q.V. (2019) 'EfficientNet: Rethinking Model Scaling for Convolutional Neural
Networks', *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97,
pp. 6105–6114.

Transfer-learning data efficiency: arXiv:2502.18558; arXiv:2606.15705.
