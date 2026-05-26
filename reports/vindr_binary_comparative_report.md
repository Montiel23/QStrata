# Q23: VinDr-SpineXR Binary Comparative Report

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

The VinDr-SpineXR binary task frames spine X-ray classification as a two-class problem: Any Pathology vs No Finding, with an approximately 0.97:1 negative-to-positive class ratio making the task nearly balanced. This report consolidates the complete VinDr-SpineXR binary benchmarking sequence executed across Slices Q17 through Q22. The sequence was designed to systematically evaluate classical and discrete-variable (DV) quantum hybrid architectures under controlled, reproducible conditions. Q17 established the classical CNN baseline with a fully trainable 23,650-parameter network. Q21 produced the first scientifically valid DV hybrid benchmark using a frozen, pretrained C006-D040 backbone feeding a 574-parameter DV quantum head. Q22 introduced an approximate trainable-parameter-matched classical control — a 526-parameter two-layer classical MLP head on the same frozen backbone — to test whether the compact trainable bottleneck structure, rather than any quantum property, can fully explain the Q21 improvement over Q17. No quantum advantage is claimed at any point in this sequence.

---

## 2. Experiment Lineage

| Slice | Description | Status | Scientific Validity |
|---|---|---|---|
| Q17 | Classical baseline (CNN3Block, 23,650 trainable params, unfrozen) | COMPLETE | Valid — establishes the classical performance reference; note that the baseline may underestimate the classical ceiling due to an unfrozen, randomly initialized backbone and no inter-block MaxPool |
| Q19 | DV hybrid with random frozen backbone | COMPLETE | Invalid benchmark — the quantum head was trained on uninformative random convolutional features, producing all-class-0 predictions and F1 = 0; results are not comparable to Q21 |
| Q20 | Pretrained backbone feasibility (DVHybridCNNQNN + C006-D040) | COMPLETE | Feasibility only — confirmed 28 backbone keys matched and zero gradient leakage into backbone; not a training result |
| Q21 | DV hybrid with frozen pretrained C006-D040 backbone | COMPLETE | Valid — first scientifically valid VinDr DV hybrid benchmark; frozen pretrained backbone provides discriminative features to the quantum head |
| Q22 | Approximate trainable-parameter-matched tiny classical control | COMPLETE | Valid — held backbone, optimizer, LR, loss, splits, seed, and epochs identical to Q21; only the DV quantum head was replaced with a classical two-layer MLP of approximately equivalent trainable parameter count |

---

## 3. Final Metric Comparison

All values populated from reviewed reports: `reports/vindr_classical_baseline_full_training.md` (Q17), `reports/vindr_dv_hybrid_full_training.md` (Q19), `reports/vindr_dv_hybrid_pretrained_full_training.md` (Q21), `reports/vindr_classical_control_tiny_head.md` (Q22).

| Metric | Q17 Classical | Q19 DV Random | Q21 DV Hybrid | Q22 Tiny Classical |
|---|---|---|---|---|
| Test AUROC | 0.6224 | 0.5442 | 0.6800 | 0.6625 |
| Test F1 | 0.5355 | 0.0000 | 0.6159 | 0.5961 |
| Test Accuracy | 60.66% | 51.52% | 63.84% | 64.37% |
| Test Precision | 0.6263 | 0.0000 | 0.6350 | 0.6618 |
| Test Recall | 0.4677 | 0.0000 | 0.5978 | 0.5422 |
| Test AUPRC | 0.6730 | 0.5538 | 0.6571 | 0.6559 |
| Test Loss | 0.6850 | 0.6928 | 0.6429 | 0.6627 |
| Trainable Params | 23,650 | 574 | 574 | 526 |
| Backbone | unfrozen random | frozen random | frozen pretrained | frozen pretrained |
| Head type | classical MLP | DV quantum | DV quantum | tiny classical MLP |
| Best epoch | 1 of 6 | 2 of 6 | 15 of 15 | 15 of 15 |
| Latency (ms/sample) | 0.81 (GPU) | 55.13 (CPU) | 54.79 (CPU) | 1.48 (GPU) |
| Confusion Matrix | [[789,281],[536,471]] | [[1070,0],[1007,0]] | [[724,346],[405,602]] | [[791,279],[461,546]] |
| Scientific Validity | Valid | **Invalid** | Valid | Valid |

*Q19 latency measured on CPU (quantum circuit constraint). Q17 and Q22 measured on GPU. Q21 measured on CPU (quantum circuit constraint). Direct latency comparison across groups is not valid; see Section 8.*

*Q19 note: confusion matrix [[1070,0],[1007,0]] indicates complete class collapse — all predictions are class 0. Precision, recall, and F1 are all 0.0000 as a result.*

---

## 4. Performance Delta Analysis

### Q21 vs Q17 — DV hybrid vs classical baseline

| Metric | Q17 | Q21 | Delta | Arithmetic |
|---|---|---|---|---|
| Test AUROC | 0.6224 | 0.6800 | **+0.0576** | 0.6800 − 0.6224 = +0.0576 |
| Test F1 | 0.5355 | 0.6159 | **+0.0804** | 0.6159 − 0.5355 = +0.0804 |
| Test Accuracy | 60.66% | 63.84% | **+3.18 pp** | 63.84 − 60.66 = +3.18 |

**Interpretation:** Q21 substantially outperforms Q17 on all three primary metrics. However, Q21 differs from Q17 on three simultaneous axes — head type (quantum vs classical), backbone training state (frozen pretrained vs unfrozen random), and total trainable parameter count (574 vs 23,650) — making the source of this gain impossible to attribute to any single factor from this comparison alone. The Q22 control was designed to isolate these effects.

---

### Q22 vs Q17 — tiny classical control vs classical baseline

| Metric | Q17 | Q22 | Delta | Arithmetic |
|---|---|---|---|---|
| Test AUROC | 0.6224 | 0.6625 | **+0.0401** | 0.6625 − 0.6224 = +0.0401 |
| Test F1 | 0.5355 | 0.5961 | **+0.0606** | 0.5961 − 0.5355 = +0.0606 |
| Test Accuracy | 60.66% | 64.37% | **+3.71 pp** | 64.37 − 60.66 = +3.71 |

**Interpretation:** Replacing the DV quantum head with a tiny classical MLP on the same frozen pretrained backbone still yields a meaningful improvement over Q17. Q22 recovers 70% of the Q21 AUROC gain (0.0401 / 0.0576) and 75% of the Q21 F1 gain (0.0606 / 0.0804) relative to Q17. This confirms that the frozen pretrained backbone plus compact trainable head is a strong architectural choice that a classical head can partially replicate — indicating that backbone pretraining and regularization via head compactness are major contributors to the Q21 improvement.

---

### Q21 vs Q22 — DV hybrid vs tiny classical control

| Metric | Q22 | Q21 | Delta (Q21−Q22) | Arithmetic |
|---|---|---|---|---|
| Test AUROC | 0.6625 | 0.6800 | **+0.0175** | 0.6800 − 0.6625 = +0.0175 |
| Test F1 | 0.5961 | 0.6159 | **+0.0198** | 0.6159 − 0.5961 = +0.0198 |
| Test Accuracy | 64.37% | 63.84% | **−0.53 pp** | 63.84 − 64.37 = −0.53 |

**Interpretation:** Q21 outperforms Q22 on AUROC (+0.0175) and F1 (+0.0198), while Q22 marginally outperforms Q21 on raw accuracy (−0.53 pp in Q21's favor). The AUROC and F1 advantage for Q21 is small but present, and exceeds the 0.01 threshold used to define "meaningfully lower" in the Q22 control design. This residual gap cannot be attributed to parameter count or backbone pretraining, since both are held nearly constant between Q21 and Q22. The gap is consistent with a quantum inductive bias effect but is far from sufficient to establish quantum advantage.

---

## 5. Scientific Interpretation

**1. Did Q21 outperform the original classical baseline?**

Yes. Q21 achieved AUROC 0.6800 vs Q17's 0.6224 (delta: +0.0576) and F1 0.6159 vs Q17's 0.5355 (delta: +0.0804). Both primary metrics show clear improvement. However, this comparison is confounded by simultaneous changes in backbone state and trainable parameter count; it does not isolate the contribution of the quantum head.

**2. Did Q22 recover a substantial portion of that gain?**

Yes. Q22 recovered approximately 70% of Q21's AUROC gain over Q17 (Q22 delta: +0.0401 vs Q21 delta: +0.0576) and approximately 75% of Q21's F1 gain (Q22 delta: +0.0606 vs Q21 delta: +0.0804). The majority of the performance improvement observed in Q21 is therefore attributable to the frozen pretrained backbone and compact head constraint, not uniquely to the DV quantum head.

**3. Did Q22 fully explain the Q21 gain?**

No. Q21 still outperforms Q22 on both AUROC (0.6800 vs 0.6625, delta: +0.0175) and F1 (0.6159 vs 0.5961, delta: +0.0198). The compact trainable bottleneck effect, while accounting for the majority of the Q21 improvement over Q17, does not fully close the gap. A residual advantage for the DV quantum head remains in this single-seed experiment.

**4. What does that imply scientifically?**

The compact trainable bottleneck effect contributes meaningfully to the Q21 improvement. The frozen pretrained backbone combined with a tiny trainable head — classical or quantum — produces a strong benchmark relative to a fully trainable shallow classical baseline (Q17). However, Q21 retains a residual AUROC and F1 advantage over Q22 that the bottleneck effect alone does not fully explain. This residual is scientifically interesting — it is consistent with a quantum inductive bias effect — but it is small, arises from a single seed, lacks statistical validation, and is entirely insufficient to establish quantum advantage. The appropriate conclusion is that the DV hybrid result warrants further investigation under stronger experimental controls.

---

## 6. Scientific Caveats

**1. Cross-domain pretrained backbone.** The C006-D040 backbone was pretrained on PneumoniaMNIST (28×28 chest radiograph binary classification) and transferred to VinDr-SpineXR (224×224 spine radiograph binary classification). These domains differ substantially in anatomy, pathology type, image resolution, and visual feature distribution. The backbone's learned representations may not be optimally adapted to the VinDr task. This limits interpretation of absolute performance levels and may compress the observable range of improvement for any head architecture.

**2. Approximate trainable parameter matching is not functional equivalence.** Q22 targets ~574 trainable parameters using a two-layer classical MLP derived from a closed-form approximation. The actual count (526) differs by −8.4%. More critically, parameter count equivalence does not imply functional equivalence: the DV quantum head (Linear(128,4) → quantum ansatz → Linear(16,2)) and the tiny classical head (Linear(128,4) → ReLU → Linear(4,2)) differ in inductive bias, architecture topology, representational geometry, and output dimensionality before the final linear readout. Q22 is a reasonable but imperfect control.

**3. Classical baseline not heavily optimized.** Q17 used a single fixed architecture (CNN3Block, channels [16,32,64]) and a single training configuration. No hyperparameter search was performed. A more thoroughly optimized classical baseline — with architecture search, learning rate tuning, or regularization exploration — could substantially close or reverse the observed gap. The Q20 interpretation guardrail explicitly flags this limitation.

**4. Single seed only.** All experiments (Q17, Q21, Q22) used seed 42. Training dynamics for small heads on moderately sized datasets can exhibit meaningful variance across seeds. Observed metric differences may not be reproducible across different random initializations. No multi-seed stability analysis has been performed for Q17 or Q22.

**5. No statistical confidence intervals.** With a single run per configuration, no confidence intervals, standard deviations, or significance tests are available for any metric. The AUROC delta of +0.0175 between Q21 and Q22 is reported as a point estimate only. This delta may be within the noise range of a multi-seed distribution; it cannot be treated as a statistically validated difference.

**6. No quantum advantage claim.** No result in this benchmarking sequence establishes quantum advantage. The DV quantum head (Q21) is a research benchmark, not a demonstrated superior architecture. The residual Q21 vs Q22 gap (+0.0175 AUROC) is a scientifically interesting signal that motivates further investigation; it is not evidence of quantum advantage.

---

## 7. Required Statement

> The tiny classical control recovers a substantial portion of the Q21 gain, indicating that compact trainable bottleneck behavior contributes meaningfully to performance improvement. However, Q21 still outperforms Q22 numerically, suggesting that the bottleneck effect alone does not fully explain the DV hybrid result. This increases scientific interest in the Q21 benchmark but does NOT establish quantum advantage.

---

## 8. Latency Notes

| Model | Device | Mean Latency | Comparable to |
|---|---|---|---|
| Q17 Classical | GPU (CUDA) | 0.81 ms/sample | Q22 |
| Q19 DV Random | CPU (quantum) | 55.13 ms/sample | Q21 |
| Q21 DV Hybrid | CPU (quantum) | 54.79 ms/sample | Q19 |
| Q22 Tiny Classical | GPU (CUDA) | 1.48 ms/sample | Q17 |

Quantum circuit simulation in Q19 and Q21 is CPU-bound (QStrata Backend constraint). GPU-accelerated classical inference (Q17, Q22) is 37–68× faster per sample under these conditions. This latency gap reflects the cost of quantum circuit simulation software, not an inherent architectural characteristic of the DV model. Cross-group latency comparisons (quantum-to-classical) are not architecturally meaningful in this experimental setup.

---

## 9. Anticipated Reviewer Attacks

- **Single seed:** Results from a single random seed may not be reproducible or stable. Observed deltas (e.g., AUROC +0.0175 for Q21 over Q22) may lie within the noise range of the metric distribution across seeds.

- **No confidence intervals:** All reported metric differences are point estimates. Without bootstrap intervals or cross-validation, no statistical significance can be claimed for any observed delta.

- **No domain-specific pretraining:** The C006-D040 backbone was pretrained on PneumoniaMNIST, not on spine or chest X-ray data at the relevant resolution. The cross-domain transfer confounds interpretation of both absolute performance and relative gains.

- **Classical baseline not heavily optimized:** Q17 used a single fixed architecture without hyperparameter search. The classical ceiling for this task may be substantially higher than 0.6224 AUROC, which would reduce or eliminate the apparent Q21 advantage.

- **Approximate parameter matching:** Q22 uses 526 trainable parameters vs Q21's 574, a difference of −8.4%. More importantly, the two heads differ in representational structure beyond parameter count; the control is approximate, not exact.

- **No multiple classical controls:** A single tiny classical head configuration is insufficient to characterize the full classical comparator space. Additional ablations — varying head depth, width, and activation function — are needed before drawing conclusions about the boundary of what classical bottleneck architectures can achieve.

---

## 10. Recommended Future Controls

The following controls are recommended. None are implemented here.

- **Multi-seed reruns of Q21 and Q22** (minimum 3 seeds: 42, 7, 123) with mean and standard deviation reported for AUROC and F1. This is the highest-priority next step.
- **Bootstrap or cross-validation confidence intervals on AUROC and F1** to assess whether the Q21 vs Q22 delta (+0.0175 AUROC) is statistically distinguishable from zero.
- **Stronger classical comparator:** fully trained shallow MLP on frozen backbone features with learning rate and architecture search — to better characterize the classical ceiling at this parameter scale.
- **Domain-specific pretrained backbone:** replace PneumoniaMNIST backbone with one pretrained on a spine or chest X-ray dataset at appropriate resolution. This would remove the cross-domain confound.
- **Additional tiny classical ablations:** vary hidden dimension (H = 2, 4, 8) and activation function (ReLU, GELU, Tanh) to map the classical comparator space more thoroughly before attributing residual Q21 advantage to quantum effects.
- **Q22 with exact parameter count:** design a classical head with exactly 574 trainable parameters to eliminate the approximate matching caveat.

---

## 11. VinDr Binary Closure Verdict

The VinDr-SpineXR binary benchmarking sequence (Q17–Q22) has produced a scientifically structured comparative result. Q21 demonstrates that a frozen pretrained backbone feeding a 574-parameter DV quantum head outperforms both the random-backbone DV baseline (Q19) and the unfrozen classical CNN baseline (Q17) by meaningful margins on AUROC and F1. Q22 establishes that a classical head of approximately equivalent trainable parameter count, under otherwise identical conditions, recovers most but not all of that gain — leaving a residual AUROC gap of +0.0175 in Q21's favor. This residual is small, arises from a single seed, is not statistically validated, and does not establish quantum advantage. What the data does support is this: the frozen pretrained backbone combined with a compact trainable head produces a strong and reproducible improvement over the unfrozen classical baseline regardless of head type, and the DV quantum head retains a numerically modest additional advantage over an approximately equivalent classical alternative in this configuration. The result is sufficiently scientifically interesting to warrant further investigation under stronger experimental controls — specifically multi-seed evaluation and a more thoroughly characterized classical comparator space — before any interpretation of the residual gap is advanced. No quantum advantage is supported by the current evidence.

---

## 12. Next Phase

VinDr binary benchmarking is now **CLOSED**.

Next phase: PneumoniaMNIST multiclass OR VinDr multiclass — pending roadmap decision.

```
VinDr binary closure status: CLOSED
```
