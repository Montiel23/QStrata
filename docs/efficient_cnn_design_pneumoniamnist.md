# Efficient CNN Design Spec — PneumoniaMNIST

## 1. Current Baseline Result

Config-driven baseline from Slice 9 (`experiments/configs/binary_baseline.yaml`).

| Metric | Value |
|---|---|
| Conv channels | [32, 64] |
| BatchNorm | enabled |
| Dropout | 0.3 |
| Class weights | enabled |
| Total parameters | 19,138 |
| Epochs | 10 |
| Mean epoch time | 0.48 s |
| Val accuracy | 92.56% |
| Test accuracy | 86.22% |
| Inference latency | 0.60 ms / batch |
| Status | **GO** |

---

## 2. Why the Stronger Baseline Worked

The 98-parameter tiny CNN from Slices 3–4 failed to reach the 70% test accuracy threshold. The 19,138-parameter baseline from Slice 9 cleared 86% test accuracy. Four factors explain the gap:

- **Representational capacity.** A single Conv2d(1→8) layer cannot capture the spatial texture and intensity patterns that distinguish pneumonia from normal lung fields. Two stacked blocks (32→64 channels) provide sufficient depth for low-resolution 28×28 grayscale inputs.
- **BatchNorm stabilisation.** Batch normalisation after each Conv2d reduces internal covariate shift, enabling stable gradient flow and faster convergence, especially relevant for a small batch of 64 over a moderate dataset of 4,708 training samples.
- **Dropout regularisation.** Dropout(0.3) before the final Linear layer prevents the classifier head from memorising the training distribution. Without it the tiny CNN converged to a majority-class predictor.
- **Class weight correction.** PneumoniaMNIST is imbalanced (pneumonia class is the majority). Passing inverse-frequency class weights to CrossEntropyLoss forces the model to penalise minority-class errors more heavily, lifting recall on the normal class and improving overall test accuracy.

---

## 3. Efficient CNN Design Goals

| Goal | Target |
|---|---|
| Minimum test accuracy | ≥ 86% (match baseline) |
| Target test accuracy | ≥ 90% |
| Maximum parameter count | < 19,138 (reduce where possible) |
| Hard parameter ceiling | 50,000 |
| Inference latency | minimise (GPU ms/batch) |
| Epoch time | minimise |
| Validation environment | GPU Docker only |

The primary question is whether more parameter-efficient operations can match or exceed the standard conv baseline, and whether increased depth with fewer parameters per layer can push test accuracy above 90%.

---

## 4. Candidate Operations

### a. Standard Conv Block

Structure: Conv2d → BatchNorm2d → ReLU

Reference implementation. Each layer has `out_channels × in_channels × k × k` parameters. Accurate but parameter-heavy relative to the receptive field it buys on 28×28 inputs. Serves as the performance ceiling against which all efficient variants are measured.

### b. Depthwise Separable Conv Block

Structure: DepthwiseConv2d(groups=in_channels) → BatchNorm2d → ReLU → PointwiseConv2d(1×1) → BatchNorm2d → ReLU

MobileNet-style factorisation. Spatial filtering (depthwise) and channel mixing (pointwise) are decoupled. Parameter reduction factor versus standard conv: approximately `1/out_channels + 1/k²`. For k=3 this is roughly 8–9× fewer parameters at the same channel width. Expected accuracy penalty is small for simple classification tasks on low-resolution inputs.

### c. Asymmetric Conv Block

Structure: Conv2d(k×1) → Conv2d(1×k) → BatchNorm2d → ReLU

Factorises a k×k convolution into a horizontal and a vertical filter. Parameter count is `2 × channels² × k` versus `channels² × k²` for the full conv. Effective for k=3 (33% reduction) and k=5 (60% reduction). Captures anisotropic spatial patterns. Useful on chest X-rays where horizontal and vertical structure (ribs, diaphragm) are meaningful.

### d. Grouped Conv + Channel Shuffle Block

Structure: GroupedConv2d(groups=g) → BatchNorm2d → ReLU → ChannelShuffle

ShuffleNet-style. Grouped convolution reduces cross-channel compute by factor g; channel shuffle restores cross-group information flow that grouped conv would otherwise block. Parameter reduction: `1/g` versus standard conv. Requires `num_channels % num_groups == 0`. Less natural on small channel counts (32, 64) — constraint must be validated per config.

### e. Dilated Conv Block (optional)

Structure: DilatedConv2d(dilation=2) → BatchNorm2d → ReLU

Same parameter count as a standard conv with the same k. Expands effective receptive field by inserting gaps between filter elements (effective receptive field = k + (k-1)×(dilation-1)). No additional parameters. On 28×28 inputs the benefit is limited — receptive field is already global after AdaptiveAvgPool2d — but dilated conv may help capture multi-scale texture before pooling.

---

## 5. Candidate Block Config Schema (future YAML)

Extension of `experiments/configs/binary_baseline.yaml`. Fields below will be consumed by `build_model()` in `qcore/models/cnn.py` once Slice 11 is complete.

| Field | Type | Values | Default |
|---|---|---|---|
| `block_type` | string | `standard`, `depthwise_sep`, `asymmetric`, `grouped_shuffle`, `dilated` | `standard` |
| `conv_channels` | list[int] | e.g. `[32, 64]` | — |
| `kernel_size` | int | 3, 5 | 3 |
| `dilation` | int | 1, 2 | 1 |
| `groups` | int | 1, 2, 4 | 1 |
| `use_batchnorm` | bool | true, false | true |
| `dropout` | float | 0.0 – 0.5 | 0.3 |
| `pooling` | string | `adaptive_avg` | `adaptive_avg` |

`build_model()` in `qcore/models/cnn.py` will consume this schema in a future slice. The `dataset` and `training` sections of the YAML are unchanged.

---

## 6. Metrics

All experiments must report the following. No result is accepted without the full set.

| Metric | Description |
|---|---|
| Val accuracy | Accuracy on the 524-sample validation split after final epoch |
| Test accuracy | Accuracy on the 624-sample test split (held out until evaluation) |
| Total parameters | Sum of all trainable parameters |
| Inference latency | Mean ms per batch over the test DataLoader, GPU, no warmup exclusion |
| Mean epoch time | Mean wall time per training epoch (s) |

Comparison against baseline is required for every new config. Regression is defined as any config with test accuracy below 86.22%.

---

## 7. Constraints

| Constraint | Limit | Enforcement |
|---|---|---|
| Max parameters | 50,000 | Hard — configs exceeding this are not run |
| Min test accuracy | ≥ 86.22% | Match baseline — regression is a blocker |
| Target test accuracy | ≥ 90% | Design goal |
| Accuracy regression | Not permitted | Any config below baseline is discarded |
| Validation environment | GPU Docker only | No host or CPU Docker results accepted |
| Training budget | 10 epochs | Fixed to match baseline comparison |
| Augmentation | None | Deferred — not part of efficient block evaluation |

---

## 8. Proposed Next Implementation Slices

| Slice | Scope |
|---|---|
| **Slice 11** | Extend `build_model()` in `qcore/models/cnn.py` to support `block_type` field; implement all five block types |
| **Slice 12** | Add efficient block YAML configs to `experiments/configs/`: one config per block type at matched channel width |
| **Slice 13** | Run manual experiments for each config; collect full metrics table; compare to baseline |
| **Slice 14** | Validate best-performing efficient config; must meet ≥ 86% test accuracy before any NAS work begins |
| **Slice 15** | Prepare pymoo NSGA-II search space using validated block types only |

---

## 9. Hard Boundaries

- No JetSeg architecture copy-paste — PneumoniaMNIST is a classification task; no segmentation decoder is needed or permitted.
- No segmentation decoder.
- No JetLoss.
- No NAS implementation until efficient blocks are manually validated (Slices 11–14 complete).
- No pymoo until Slice 15.
