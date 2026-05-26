# Q28: DV vs CV Binary Comparative Report

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

This report completes the VinDr-SpineXR binary quantum benchmarking phase by providing the first formal comparison of all four evaluated model types under compact parameter budgets: the classical CNN baseline (Q17), the discrete-variable (DV) quantum hybrid (Q21), the approximate trainable-parameter-matched classical control (Q22), and the continuous-variable (CV) quantum hybrid (Q27).

The benchmarking sequence proceeded in strict scientific order:

- **Q17** established the unfrozen classical CNN baseline using a 23,650-parameter CNN3Block architecture trained from random initialization on VinDr-SpineXR binary classification (Any Pathology vs No Finding). This provided the performance reference for all subsequent comparisons.
- **Q21** produced the first scientifically valid DV hybrid benchmark: a frozen, pretrained C006-D040 backbone (PneumoniaMNIST-pretrained) feeding a 574-parameter DV quantum head (projection, 4-qubit variational ansatz, linear readout).
- **Q22** introduced a necessary scientific control: the same frozen pretrained backbone feeding a tiny classical MLP head (526 trainable parameters), matched approximately to Q21's parameter count. This isolated whether compact bottleneck behavior, independent of the quantum head, explains the Q21 improvement.
- **Q26** validated the CV pipeline numerically: forward pass, gradient flow, covariance health, and optimizer update — all 14 health checks PASS.
- **Q27** produced the first CV hybrid benchmark using the exact same frozen pretrained backbone, a CV Gaussian head (compression → GaussianVariationalAnsatz → deterministic first-moment readout → linear classifier, 536 trainable parameters total).

Q28 is the first report that simultaneously compares DV hybrid, CV hybrid, and classical controls side by side under compact parameter budgets on the VinDr-SpineXR binary task.

All values in this report are drawn exclusively from the source reports. No values are inferred, estimated, or filled from memory.

**Source reports:**
- `reports/vindr_classical_baseline_full_training.md` (Q17)
- `reports/vindr_dv_hybrid_pretrained_full_training.md` (Q21)
- `reports/vindr_classical_control_tiny_head.md` (Q22)
- `reports/q27_cv_binary_full_training.md` (Q27)
- `reports/vindr_binary_comparative_report.md` (Q23 — DV closure report)
- `reports/q27a_nas_strategy_and_optimization_refinement.md` (Q27A — roadmap refinement)

---

## 2. Experimental Conditions

All four models were evaluated on the same VinDr-SpineXR binary dataset split, with the following shared conditions:

| Condition | Value |
|---|---|
| Dataset | `data/processed/vindr_binary_roi_224` |
| Task | Binary: Any Pathology (1) vs No Finding (0) |
| Seed | 42 (single seed) |
| Loss | Unweighted CrossEntropyLoss |
| Optimizer | AdamW, lr=1e-3 |
| Evaluation set | Same test split for all four models |
| Test set size | 2,077 images (1,070 class 0, 1,007 class 1) |
| Backbone checkpoint (Q21/Q22/Q27) | `checkpoints/c006_d040_classical_anchor.pt` (C006-D040 PneumoniaMNIST-pretrained) |

**Key difference:** Q17 uses an unfrozen randomly initialized backbone (23,650 params, full end-to-end training). Q21, Q22, and Q27 all use the same frozen pretrained backbone with a compact trainable head (≈526–574 trainable params).

---

## 3. Benchmark Summary

All values extracted from source reports. Sources cited per column.

### Table 1 — Main Benchmark Table

| Model | Type | AUROC | F1 | Accuracy | Precision | Recall | AUPRC | Params (trainable) | Latency |
|---|---|---|---|---|---|---|---|---|---|
| Q17 | Classical CNN | 0.6224 | 0.5355 | 60.66% | 0.6263 | 0.4677 | 0.6730 | 23,650 | 0.8114 ms/sample (GPU) |
| Q21 | DV Hybrid | 0.6800 | 0.6159 | 63.84% | 0.6350 | 0.5978 | 0.6571 | 574 | 54.7855 ms/sample (CPU) |
| Q22 | Tiny Classical Control | 0.6625 | 0.5961 | 64.37% | 0.6618 | 0.5422 | 0.6559 | 526 | 1.48 ms/sample (GPU) |
| Q27 | CV Hybrid | 0.6708 | 0.6283 | 65.77% | 0.6634 | 0.5968 | 0.6560 | 536 | 2.1538 ms/sample (CUDA+CPU) |

*Q17 latency: GPU (CUDA), 100 timed forward passes, batch size 16 (report Section 12). Q21 latency: CPU, 30 timed forward passes, batch size 4, quantum circuit CPU-bound (report Section 12). Q22 latency: GPU (CUDA), 100 timed single-sample passes (report Section 11). Q27 latency: CUDA backbone + CPU CV backend, 100 timed single-sample passes (report Section 12). Direct cross-model latency comparison is not architecturally valid due to different device assignments; see Section 7.*

### Table 2 — Confusion Matrices

Labels: 0 = No Finding (negative), 1 = Any Pathology (positive).

| Model | TN | FP | FN | TP | Recall | Precision |
|---|---|---|---|---|---|---|
| Q17 Classical | 789 | 281 | 536 | 471 | 471/(471+536) = 0.4677 | 471/(471+281) = 0.6263 |
| Q21 DV Hybrid | 724 | 346 | 405 | 602 | 602/(602+405) = 0.5978 | 602/(602+346) = 0.6350 |
| Q22 Tiny Classical | 791 | 279 | 461 | 546 | 546/(546+461) = 0.5422 | 546/(546+279) = 0.6618 |
| Q27 CV Hybrid | 765 | 305 | 406 | 601 | 601/(601+406) = 0.5967 | 601/(601+305) = 0.6634 |

All four confusion matrices are non-degenerate: all cells (TN, FP, FN, TP) are nonzero for all models. No class collapse was observed.

---

## 4. Compactness Analysis

Q21, Q22, and Q27 each use between 526 and 574 trainable parameters — a reduction of approximately 97.6% relative to Q17's 23,650 trainable parameters. The trainable parameter ratios are:

| Model | Params | Ratio vs Q17 |
|---|---|---|
| Q17 | 23,650 | 1.00 (reference) |
| Q21 | 574 | 574 / 23,650 = 0.024 |
| Q22 | 526 | 526 / 23,650 = 0.022 |
| Q27 | 536 | 536 / 23,650 = 0.023 |

Despite using approximately 2.4% as many trainable parameters as Q17, all three compact models (Q21, Q22, Q27) achieve higher test AUROC and F1 than Q17. This is not primarily a consequence of the quantum heads: Q22, using a simple two-layer classical MLP head, also surpasses Q17 on AUROC (+0.0401) and F1 (+0.0606). The dominant driver of this improvement is the frozen pretrained backbone, not the head architecture.

**Frozen pretrained backbone effect.** The C006-D040 backbone was pretrained on PneumoniaMNIST (28×28 grayscale chest X-ray classification, slice Q6). When frozen and applied as a feature extractor for VinDr-SpineXR (224×224 spine X-ray binary classification), it provides fixed feature representations that the compact head maps to the target task. Because the backbone is not fine-tuned, there is no risk of gradient-driven destabilization of the feature extractor on the smaller VinDr dataset. The feature representations provide a stable signal for the head to learn a minimal decision boundary.

**Compact representation learning.** Forcing the training signal through a very small trainable module (526–574 parameters) acts as a strong inductive regularizer. The head cannot overfit a large set of spurious correlations; it must compress the 128-dimensional backbone output into a binary classification decision using fewer degrees of freedom than a classical baseline with 23,650 parameters. This is a classical transfer learning effect that applies to any head architecture — quantum or otherwise.

**Cross-domain transfer caveat.** The backbone was pretrained on a different anatomical domain (chest radiograph, PneumoniaMNIST) at a different resolution (28×28) than the target domain (spine radiograph, VinDr-SpineXR, 224×224). While the backbone empirically provides useful features for the VinDr task, its representations are not domain-optimized. Absolute performance levels and the relative ranking of head architectures may change with a domain-specific pretrained backbone. This is a known limitation of the current benchmarking setup (see Section 9).

**Important accounting note.** The frozen backbone contributes 9,612 parameters to total model capacity in Q21, Q22, and Q27. These parameters are not counted as trainable — they do not receive gradient updates during the VinDr training run — but they are a significant component of the model's representational power. The compactness claim refers specifically to the number of parameters adapted to the VinDr task, not to total model size.

---

## 5. Classical Control Interpretation

Q22 was designed with a specific scientific purpose: to test whether compact bottleneck behavior alone — independent of any quantum property — can explain the Q21 improvement over Q17. This is a necessary control before drawing conclusions about the DV hybrid architecture.

**Q22 design.** The Q22 head replaces the DV quantum circuit (projection → quantum ansatz → readout, 574 params total) with a two-layer classical MLP (Linear(128,4) → ReLU → Linear(4,2), 526 params total). All other experimental conditions are held constant: same frozen pretrained backbone, same optimizer (AdamW lr=1e-3), same loss (unweighted CrossEntropyLoss), same batch size (4), same max epochs (15), same early stopping patience (4), same seed (42), same dataset split.

**What Q22 found.** Q22 achieved test AUROC 0.6625, recovering a substantial portion of the Q21 gain over Q17:

| Gain | AUROC | F1 |
|---|---|---|
| Q21 vs Q17 (full compact-hybrid gain) | +0.0576 | +0.0804 |
| Q22 vs Q17 (compact-classical gain) | +0.0401 | +0.0606 |
| Fraction of Q21 gain recovered by Q22 | 0.0401 / 0.0576 = 70% | 0.0606 / 0.0804 = 75% |

Q22 recovers approximately 70–75% of the Q21 gain over Q17 using a classical head. This demonstrates that compact bottleneck behavior is the dominant contributor to the improvement observed in Q21 over Q17. The frozen pretrained backbone combined with a tiny trainable head — regardless of whether the head is quantum or classical — yields a strong improvement over the unfrozen classical baseline (Q17).

**Residual gap.** Q21 still outperforms Q22 by AUROC +0.0175 and F1 +0.0198. This residual is not explained by parameter count or backbone pretraining (both are approximately held constant). It is consistent with a quantum inductive bias effect in the DV head, but it is small, arises from a single seed, and has no statistical validation.

**Q27 in this context.** Q27 (CV hybrid, 536 params) achieves AUROC 0.6708, placing it between Q22 (0.6625) and Q21 (0.6800). Q27 also exceeds Q22 on AUROC by +0.0083 and on F1 by +0.0322. This pattern is consistent with the picture established by Q21 and Q22: compact hybrid quantum heads exceed the parameter-matched classical control on AUROC and F1, but by small margins that do not support strong conclusions.

The parameter-matched classical control (Q22) is essential for interpreting any compact quantum hybrid result. Without Q22, the Q21 and Q27 improvements over Q17 could plausibly be attributed entirely to the compact bottleneck and frozen backbone, with no role for the quantum head.

---

## 6. DV vs CV Comparative Analysis

### Table 3 — Delta Table

All deltas computed from source report values.

| Comparison | AUROC Delta | F1 Delta | Param Ratio |
|---|---|---|---|
| Q21 vs Q17 | 0.6800 − 0.6224 = **+0.0576** | 0.6159 − 0.5355 = **+0.0804** | 574 / 23,650 = 0.024 |
| Q22 vs Q17 | 0.6625 − 0.6224 = **+0.0401** | 0.5961 − 0.5355 = **+0.0606** | 526 / 23,650 = 0.022 |
| Q27 vs Q17 | 0.6708 − 0.6224 = **+0.0484** | 0.6283 − 0.5355 = **+0.0928** | 536 / 23,650 = 0.023 |
| Q21 vs Q22 | 0.6800 − 0.6625 = **+0.0175** | 0.6159 − 0.5961 = **+0.0198** | 574 / 526 = 1.09 |
| Q27 vs Q22 | 0.6708 − 0.6625 = **+0.0083** | 0.6283 − 0.5961 = **+0.0322** | 536 / 526 = 1.02 |
| Q21 vs Q27 | 0.6800 − 0.6708 = **+0.0092** | 0.6159 − 0.6283 = **−0.0124** | 574 / 536 = 1.07 |

### DV vs CV Pattern

**AUROC.** DV (Q21) achieves AUROC 0.6800 and CV (Q27) achieves AUROC 0.6708 — a difference of +0.0092 in favor of DV. Both exceed the classical control (Q22, AUROC 0.6625).

**F1.** CV (Q27) achieves F1 0.6283 and DV (Q21) achieves F1 0.6159 — a difference of +0.0124 in favor of CV. Both exceed the classical control (Q22, F1 0.5961).

**Accuracy.** CV (Q27) achieves accuracy 65.77%, DV (Q21) 63.84%, and the classical control Q22 achieves 64.37%. Accuracy is a less informative discriminator for this class ratio.

**The AUROC–F1 inversion.** DV leads on AUROC while CV leads on F1. This is not contradictory. AUROC is a rank-based, threshold-independent metric that reflects the model's overall ability to discriminate between classes across all possible decision thresholds. F1 is threshold-dependent, computed at a specific operating point (default threshold 0.5), and is sensitive to the specific shape of the predicted probability distribution rather than its overall ranking quality. A model can lead on AUROC while trailing on F1 at a fixed threshold if its predicted probabilities are more spread out or better calibrated around the default threshold in a different way.

**Possible interpretations of the inversion.** The CV head's first-moment readout (`mu_final`) may produce predicted probabilities that are better calibrated near the default 0.5 threshold, leading to higher F1 at that operating point, while the DV head may provide better overall ranking separation (higher AUROC). Alternatively, the DV and CV circuit topologies may encode different inductive biases that result in different operating-point behaviors: the DV ansatz (4-qubit variational circuit with entanglement) versus the CV Gaussian circuit (2-mode displacement, squeezing, beamsplitter, rotation) differ fundamentally in how they transform input features into output representations. These interpretations are speculative: the evidence is a single-seed comparison at one operating point.

**What cannot be concluded.** The DV–CV difference (+0.0092 AUROC in favor of DV, +0.0124 F1 in favor of CV) is small in absolute terms. Both differences arise from a single-seed experiment. No confidence intervals are available. These differences are not statistically validated and may not be reproducible across different seeds, different dataset splits, or different backbone configurations. Neither DV nor CV can be declared superior based on these results.

---

## 7. Numerical Stability Analysis

The CV pipeline was validated through two stages before full training.

**Q26 smoke test (1 batch, 14 health checks — all PASS).** The following checks were confirmed before any training:

| Check | Result |
|---|---|
| FORWARD_PASS | PASS — no error |
| LOGITS_FINITE | PASS — no NaN or inf |
| MU_FINITE | PASS — no NaN or inf in first moments |
| COV_FINITE | PASS — no NaN or inf in covariance |
| COV_SYMMETRIC | PASS — \|cov − covᵀ\| < 1e−5 |
| COV_PSD | PASS — all eigenvalues ≥ −1e−6 |
| GRAD_COMPRESSION | PASS — norm = 4.1564e+00 > 0 |
| GRAD_ANSATZ | PASS — norm = 9.6710e−01 > 0 |
| GRAD_READOUT | PASS — norm = 8.2346e−01 > 0 |
| BACKBONE_FROZEN | PASS — zero backbone gradient confirmed |
| PARAMS_UPDATED | PASS — parameters changed after AdamW step |

**Q27 full training CV health (15 epochs — all PASS).** Per the Q27 report Section 11:

| Check | Result (all 15 epochs) |
|---|---|
| COV_PSD | PASS — all 15 epochs |
| COV_SYMMETRIC | PASS — all 15 epochs |
| QUAD_FINITE | PASS — all 15 epochs |
| NO_NAN_INF | PASS — all 15 epochs |

**State evolution summary** (from Q27 report Section 11):
- Mean mu magnitude: started at 0.6384, ended at 0.6383 (epoch 15). The Gaussian state is displaced non-trivially from vacuum throughout training.
- Max mu: 1.9479 at epoch 15. No divergence.
- Squeezing norm: started at 0.0640, ended at 0.1674 — reflecting active adaptation of squeezing parameters.
- Ansatz parameter norm: started at 15.5122, ended at 12.8502 — confirming active parameter updates throughout training.
- Covariance diagonal mean: ranged from 1.0091 to 1.0628 across epochs — close to the vacuum value (1.0 at hbar=2.0), no unbounded growth.
- Gradient health: all gradient norms (compression, ansatz, readout) remained finite across all 15 epochs. No gradient explosion or vanishing detected.

**Significance.** Numerical stability of the CV pipeline throughout the entire 15-epoch training run is a necessary prerequisite for any future CV scaling, NAS, or multi-seed reproducibility work. The complete PASS record across all 15 epochs and all four health checks confirms that the GaussianBackend, GaussianVariationalAnsatz, and symplectic gate sequence operate correctly under extended training.

---

## 8. Scientific Interpretation

### Table 4 — Scientific Interpretation Table

| Observation | Supported by Data? | Interpretation Limit |
|---|---|---|
| All three compact models (Q21, Q22, Q27) outperform Q17 on AUROC | Yes | Single seed; cross-domain backbone; no CI; Q17 may underestimate classical ceiling |
| Frozen pretrained backbone is the dominant contributor to compact model improvement | Yes (Q22 recovers 70–75% of Q21 gain) | Approximate parameter matching; backbone is cross-domain |
| DV hybrid has slightly higher AUROC than CV hybrid | Yes | Difference is +0.0092; single seed; not statistically validated |
| CV hybrid has slightly higher F1 than DV hybrid | Yes | Difference is +0.0124; threshold-dependent metric; single seed; not statistically validated |
| Both DV and CV hybrids exceed the classical control (Q22) on AUROC and F1 | Yes | Q21 by +0.0175 AUROC; Q27 by +0.0083 AUROC; small margins; no CI |
| Compact bottleneck contributes substantially to improvement vs Q17 | Yes (Q22 evidence) | Approximate parameter matching only; not exact functional equivalence |
| DV head provides residual benefit beyond compact bottleneck (AUROC) | Partially (Q21 > Q22) | Small delta (+0.0175); single seed; no CI; may be within noise range |
| CV head provides residual benefit beyond compact bottleneck (AUROC) | Partially (Q27 > Q22) | Small delta (+0.0083); single seed; no CI; smaller than DV residual |
| AUROC–F1 inversion between DV and CV is interpretable | Speculative | Single-seed observation; specific operating-point behavior; no threshold sweep reported |
| Quantum advantage established | No | Not supported under any definition or by any metric in this benchmarking sequence |
| Statistical superiority of any model over any other established | No | All comparisons are single-seed point estimates without CI or significance testing |
| Clinical readiness indicated | No | Not evaluated; exploratory benchmarking study only |

**What the data supports.** Under a frozen, cross-domain pretrained backbone with compact trainable heads (~500–580 trainable parameters), both DV and CV quantum hybrid models produce test AUROC and F1 above the compact classical control (Q22) and substantially above the unfrozen classical baseline (Q17). The dominant driver of improvement is the frozen pretrained backbone combined with compact head regularization, not the quantum head type. Both quantum head types (DV, CV) produce a small residual advantage over the parameter-matched classical control, but neither exceeds the noise range one would expect from multi-seed variance.

**What the data does not support.** No claim of quantum advantage, quantum superiority over classical models, or quantum inductive bias can be supported from single-seed point estimates. The DV–CV delta (+0.0092 AUROC in favor of DV) and the CV–DV F1 delta (+0.0124 in favor of CV) are below any threshold that could support architectural conclusions without confidence intervals. No causal claim about the role of the quantum circuit component is warranted.

**What remains unknown.** Multi-seed variance for Q21, Q22, and Q27; whether the DV–CV performance ranking is stable across seeds and splits; whether a domain-specific backbone changes the relative ranking; whether stronger classical baselines (Q32 NAS) close or reverse the quantum-over-classical residual gap; whether the AUROC–F1 inversion is a consistent property of DV vs CV heads or a single-run artefact.

---

## 9. Limitations

### Table 5 — Limitations Table

| Limitation | Why It Matters | Future Mitigation |
|---|---|---|
| Single seed (seed=42) for all four models | Results may not be stable; no variance estimate; observed deltas may lie within noise range | Multi-seed reruns (minimum seeds 42, 7, 123) in Q30+ optimization phase |
| No confidence intervals | All metric differences are point estimates; no significance test possible | Bootstrap CI on AUROC and F1 in future optimization phase |
| Cross-domain pretrained backbone | PneumoniaMNIST → VinDr-SpineXR; backbone not domain-optimized; absolute levels and relative rankings may change | Domain-specific backbone pretraining or VinDr-pretrained alternatives |
| Classical baseline (Q17) not heavily optimized | Q17 uses fixed architecture and single training run; classical ceiling may be substantially higher than 0.6224 AUROC | Classical NAS (Q32) to establish compact classical ceiling |
| Approximate parameter matching for Q22 | Q22 has 526 vs Q21's 574 params (−8.4%); functional equivalence is not established; head topologies differ beyond parameter count | Tighter parameter-matched controls in future ablation studies |
| No domain-adapted or fine-tuned backbone | All three compact models share the same cross-domain frozen backbone; backbone capability is not task-specific | Domain-specific backbone pretraining |
| No external validation set | Single train/val/test split; no held-out cohort from different institution or scanner | Future external cohort evaluation |
| No multiclass evaluation | Binary task only; real diagnostic spine X-ray tasks involve multiple pathology types | Multiclass phase after Q29 |
| No NAS or architecture search | Current architectures are fixed single configurations; may not be optimal | Q32–Q34 optimization phase; classical NAS before quantum NAS |
| No statistical significance testing | All comparisons are point estimates; AUROC delta +0.0092 DV vs CV has no CI | Future statistical analysis with multi-seed runs |
| Latency comparison confounded by device assignment | Q21 is CPU-bound by quantum circuit simulation; Q22 runs on GPU; direct latency comparison not valid | Standardized measurement protocol in future work |
| CV latency measurement is CUDA-backbone + CPU-CV hybrid | The 2.15 ms/sample figure reflects both CUDA backbone inference and CPU Gaussian circuit simulation; not directly comparable to any other model | Unified device assignment in future CV work |

---

## 10. Future Research Directions

**Q29 — Binary Quantum Release Tagging.** Q28 closes the comparative analysis; Q29 formally closes the binary quantum benchmarking program with release tagging. No NAS, AWS, Ray, or multiclass work begins before Q29 is complete.

**Optimization Phase (Q30–Q35) — after Q29.** The Q30–Q35 phase begins after the binary benchmarking program is formally closed:
- **Q30** designs the experiment automation framework (reproducible orchestration, metric tracking, checkpoint management).
- **Q31** implements the local GPU experiment runner.
- **Q32** establishes the classical NAS ceiling: search over compact classical CNN architectures before any quantum architecture search. The strongest compact classical baseline must be known before quantum NAS conclusions are drawn.
- **Q33** performs quantum head NAS (DV and CV search spaces) using the classical ceiling from Q32 as the evaluation reference.
- **Q34** runs a local multi-objective NAS pilot optimizing jointly across AUROC, F1, parameter count, latency, and training stability.
- **Q35** designs distributed NAS scaling only after local NAS is validated.

The classical ceiling principle governs Q32–Q33 sequencing: quantum NAS never precedes classical NAS. Without a validated compact classical ceiling, any quantum NAS result lacks a scientifically grounded comparison point.

**Multiclass Phase — after Q29.** VinDr-SpineXR multiclass classification begins only after the binary benchmarking program is formally closed (Q29). Multiclass evaluation will test generalization beyond the binary task to the full pathology taxonomy.

**Stronger classical baselines.** The Q17 classical baseline (23,650 params, single architecture, no NAS) likely underestimates the classical ceiling for this task. Q32 will characterize the strongest compact classical baseline achievable before making any claims about quantum residuals. Results in this report should not be interpreted as evidence that classical CNNs are outperformed by quantum hybrids: the classical comparator space has not been adequately explored.

**Multi-seed validation.** The highest-priority future step before any architectural conclusion is multi-seed evaluation of Q21, Q22, and Q27 (minimum 3 seeds each), with mean and standard deviation of AUROC and F1 reported. Without this, all observed deltas remain uninformative with respect to statistical significance.

The future research directions above do not pre-announce results. NAS may find that compact classical architectures fully close or reverse the observed quantum–classical gap. That outcome would itself be scientifically valuable: it would establish that quantum circuit heads provide no additional benefit over optimally compact classical heads in this task and dataset configuration.

---

## 11. Required Scientific Guardrail

> Q28 compares compact DV and CV hybrid benchmarks under the QStrata framework. These results do NOT establish quantum advantage, statistical superiority, or clinical readiness. The experiments are exploratory compact benchmarking studies under constrained parameter budgets and single-seed evaluation.

---

## 12. Closure Logic

Q28 closes the comparative interpretation of the VinDr binary quantum benchmarking phase. All four models (Q17, Q21, Q22, Q27) have been evaluated on the same binary classification task, compared across the same metric suite, and interpreted with appropriate scientific caveats. The DV binary phase was closed at Q23. The CV binary phase is interpretively closed at Q28.

Q29 (Binary Quantum Release Tagging) formally closes the binary quantum benchmarking program. The following remains blocked until Q29 is complete:
- NAS, AWS, and Ray work (Q30–Q35): BLOCKED until Q29
- Multiclass phase: BLOCKED until Q29

No automation, architecture search, or distributed infrastructure work may begin before Q29.

---

## 13. Next Slice

**Q29 — Binary Quantum Release Tagging**

Purpose: formally close the binary quantum benchmarking program with release tags for the VinDr DV binary phase (`vindr-binary-dv-v1`), the VinDr CV binary phase (`vindr-binary-cv-v1`), and the full VinDr binary quantum program (`vindr-binary-complete-v1`).

All tags are created after Q29 confirms all required binary phase slices are complete. Tags are not created before Q29.

---

```
Q28 status: COMPLETE
Q29 status: NEXT — Binary Quantum Release Tagging
NAS/AWS/Ray: BLOCKED until Q29
Multiclass: BLOCKED until Q29
```
