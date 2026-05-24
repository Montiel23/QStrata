# Experiment Design: NSGA-II NAS on PneumoniaMNIST

## 1. Research Goal

Manual architecture tuning on a binary medical imaging task produces a single solution optimised for one criterion (usually accuracy). For deployment on constrained hardware, accuracy alone is insufficient — parameter count and inference latency matter equally.

NSGA-II NAS replaces manual tuning with a multiobjective evolutionary search that produces a Pareto front: a set of architectures where no candidate is strictly better than another across all three objectives simultaneously. The human engineer then selects an operating point based on deployment requirements rather than a fixed accuracy threshold.

On PneumoniaMNIST, the goal is to determine whether a NAS-found architecture can exceed the 3-epoch baseline accuracy while also being smaller and faster, or expose the accuracy–efficiency trade-off explicitly so the trade-off can be made deliberately.

---

## 2. Dataset

**PneumoniaMNIST** — binary chest X-ray classification (Normal vs Pneumonia), derived from the Guangzhou Women and Children's Medical Center dataset via MedMNIST v2.

| Property | Value |
|---|---|
| Task | Binary classification |
| Input shape | 28 × 28 grayscale |
| Pixel dtype | float32, range [0, 1] |
| Train samples | 4,708 |
| Val samples | 524 |
| Test samples | 624 |
| Class 0 | Normal |
| Class 1 | Pneumonia |

**Existing interface** (no changes required):

```
get_dataset("pneumoniamnist", split)  →  MedMNISTDataset
TorchDatasetAdapter(ds)               →  torch.utils.data.Dataset
```

---

## 3. Baseline

**Architecture** (Slice 3/4 inline CNN):

| Layer | Config |
|---|---|
| Conv2d | in=1, out=8, kernel=3, padding=1 |
| ReLU | — |
| AdaptiveAvgPool2d | output=(1,1) |
| Flatten | — |
| Linear | in=8, out=2 |

**Measured metrics** (3 epochs, Adam lr=1e-3, batch=32, CPU):

| Metric | Value |
|---|---|
| Val accuracy (epoch 3) | 74.24% |
| Test accuracy | 62.50% |
| Parameter count | 98 |
| Inference time / batch (bs=32) | 0.18 ms |
| Mean epoch wall time | 0.31 s |

NAS candidates must match or exceed 74.24% val accuracy to be considered non-dominated. The 62.50% test accuracy gap versus val accuracy is noted as a dataset characteristic, not a NAS blocker.

---

## 4. Metrics

Three objectives are optimised simultaneously:

- **Objective 1 — Validation accuracy**: maximise (measured after 3 training epochs)
- **Objective 2 — Parameter count**: minimise (total trainable parameters)
- **Objective 3 — Inference time**: minimise (CPU milliseconds per 32-image batch, median of 50 forward passes after 10 warmup passes)

All three objectives are evaluated on the validation split. Test split is held out until the selected architecture is retrained in Slice 9.

---

## 5. Multiobjective Formulation

Each candidate architecture is encoded as a vector of discrete decision variables (see Section 6). NSGA-II evolves a population of encoded architectures, evaluating each as a triple:

```
objectives = [-val_accuracy, param_count, inference_time_ms]
```

Negating accuracy converts maximisation to minimisation, giving a uniform minimisation problem.

NSGA-II ranks candidates by Pareto dominance: candidate A dominates B if A is at least as good on all objectives and strictly better on at least one. Candidates that are not dominated by any other form the **Pareto front** — the output of the search. No single winner is declared. The human selects an operating point after inspecting the front.

Crossover and mutation act on the discrete decision variable encoding. Each new candidate is fully evaluated (3 training epochs from scratch) before entering the next generation.

---

## 6. Search Space

| Variable | Options |
|---|---|
| Number of conv layers | {1, 2, 3} |
| Filters per layer | {8, 16, 32} |
| Kernel size | {3, 5} |
| Pooling type | {AdaptiveAvgPool2d, MaxPool2d(2)} |
| Classifier head | Linear(features → 2) only |

**Fixed across all candidates:**

| Setting | Value |
|---|---|
| Input channels | 1 |
| Output classes | 2 |
| Activation | ReLU |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Training epochs per candidate | 3 |
| Hardware | CPU only |
| Data augmentation | None |

Filters-per-layer applies uniformly to all conv layers in a candidate (no per-layer filter variation in this iteration). Pooling is applied once after the final conv layer.

Theoretical search space size: 3 × 3 × 2 × 2 = 36 distinct architectural families. With population=10 and generations=5, roughly 50 evaluations sample this space.

---

## 7. Constraints

| Constraint | Limit | Rationale |
|---|---|---|
| Max parameters | 50,000 | Keeps models deployable; well above baseline (98 params) |
| Max epoch wall time | 300 s | Hard stop per candidate epoch; prevents runaway evaluations |
| Min val accuracy | > 50% | Must exceed random baseline; enforced as constraint, not objective |
| Epochs per candidate | 3 | Matches baseline; fixed to make comparisons fair |
| CUDA | Prohibited | CPU-only environment; no mixed precision |
| Augmentation | Prohibited | Deferred to post-NAS training (Slice 9 option) |

Any candidate violating the parameter or epoch time constraint is assigned a penalty fitness (val_accuracy = 0.0) and excluded from Pareto front consideration.

---

## 8. Evaluation Budget

| Parameter | Value |
|---|---|
| Population size | 10 candidates |
| Generations | 5 |
| Total evaluations | ~50 training runs |
| Epochs per run | 3 |
| Baseline epoch wall time | 0.31 s |
| Estimated time per run (baseline scale) | < 5 s |
| Estimated total wall time | TBD — observed after Slice 7 candidate trainer is built |

The baseline model trains in ~0.31 s/epoch. Larger candidates with 3 layers and 32 filters will be slower. Wall time estimate is marked TBD until the candidate trainer in Slice 7 produces per-architecture timing data.

---

## 9. Experiment Outputs

| Output | Description |
|---|---|
| Pareto front plot | 2D and 3D scatter: accuracy vs param count vs inference time; one point per non-dominated architecture |
| Non-dominated table | All Pareto-optimal candidates with val accuracy, param count, inference time, and architecture config |
| Selected architecture config | Human-chosen operating point from the front; recorded in this document after Slice 8 |
| No checkpoints | No model weights are saved during the search |

The selected architecture is carried forward to Slice 9 for full retraining and test-set evaluation against the baseline.

---

## 10. Risks

- **CPU budget**: 50 training runs at ~0.31 s/epoch baseline = low wall time for tiny models, but 3-layer 32-filter candidates may take significantly longer. Mitigated by the 300 s/epoch hard stop and population size capped at 10.
- **Shallow input**: 28×28 images limit the benefit of deeper conv stacks — a 3-layer network with MaxPool may reduce spatial dimensions too aggressively. The search space is capped at 3 layers accordingly, and the constraint check will surface any degenerate architectures.
- **Small val set**: 524 samples. Val accuracy estimates have variance; a 1% accuracy difference between candidates may not be meaningful. Noted as a limitation. Not a blocker for this experiment — the Pareto front will surface it as a flat region.
- **No augmentation**: Generalisation gap (val 74.24% vs test 62.50% on baseline) may persist for NAS candidates. Augmentation deferred to post-selection retraining in Slice 9.

---

## 11. Next Implementation Slices

| Slice | Scope |
|---|---|
| **Slice 6** | pymoo install, NSGA-II problem scaffold, decision variable encoding, no training |
| **Slice 7** | Candidate architecture builder, 3-epoch trainer, fitness evaluator; standalone smoke test |
| **Slice 8** | Full NAS run (pop=10, gen=5); Pareto front plot; non-dominated architecture table |
| **Slice 9** | Human selects architecture from front; retrain from scratch; test-set evaluation vs baseline |
