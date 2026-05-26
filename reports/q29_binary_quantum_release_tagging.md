# Q29: Binary Quantum Release Tagging

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

The VinDr-SpineXR binary quantum benchmarking phase was executed as a sequential, controlled research program across twelve slices:

- **Q17** established the classical CNN baseline (CNN3Block, 23,650 trainable parameters, unfrozen random backbone, AdamW lr=1e-3, 6 epochs to early stopping, seed 42). This was the initial performance reference for all subsequent comparisons.
- **Q21** produced the first scientifically valid DV hybrid benchmark: a frozen, pretrained C006-D040 backbone (PneumoniaMNIST-pretrained, Slice Q6) feeding a 574-parameter DV quantum head (Linear(128,4) → 4-qubit variational ansatz → Linear(16,2)). Q19, using a randomly initialized frozen backbone, was invalid and is excluded from all reference tables.
- **Q22** introduced the approximate trainable-parameter-matched classical control: the same frozen pretrained backbone feeding a 526-parameter tiny classical MLP (Linear(128,4) → ReLU → Linear(4,2)), holding all other experimental conditions constant with Q21. This control isolated the compact bottleneck effect from any quantum head contribution.
- **Q23** produced the formal VinDr DV binary comparative report: Q17 vs Q21 vs Q22 analysis with full caveats, formally closing the DV binary phase.
- **Q25** designed the CV binary experiment architecture (Gaussian ansatz, symplectic formalism, moment-based readout).
- **Q25A** planned the experiment automation roadmap and gated NAS/AWS/Ray until CV baseline validation was complete.
- **Q26** validated the CV pipeline with a one-batch smoke test: 14 health checks, all PASS.
- **Q27** produced the first CV hybrid benchmark: same frozen pretrained backbone feeding a 536-parameter CV Gaussian head (Linear(128,4) → GaussianVariationalAnsatz(n_modes=2, depth=1) → first-moment readout → Linear(4,2)), 15 full training epochs, test AUROC 0.6708.
- **Q27A** refined the Q30–Q35 automation roadmap: replaced generic placeholders with structured descriptions, added the classical ceiling principle, updated NAS gate to Q29.
- **Q28** produced the first formal DV vs CV comparative report: all four models (Q17, Q21, Q22, Q27) compared side-by-side across the full metric suite.

**Why formal closure matters.** Without annotated release tags, future experiments — optimization runs, NAS trials, multiclass baselines, ablation studies — lack stable reference anchors. A researcher who runs a new experiment six months from now needs to know exactly which commit, which script version, and which metrics define the binary baseline phase. Annotated tags on this commit create those anchors. Any future branch can recover the exact project state at binary closure by checking out these tags.

---

## 2. Context Summary — Complete Benchmarking Sequence

| Slice | Description | Status |
|---|---|---|
| Q17 | Classical CNN baseline (23,650 params, unfrozen) | COMPLETE |
| Q19 | DV hybrid with random frozen backbone | COMPLETE — scientifically invalid, excluded from reference tables |
| Q20 | Pretrained backbone feasibility check (DVHybridCNNQNN + C006-D040) | COMPLETE — feasibility only |
| Q21 | DV hybrid with frozen pretrained C006-D040 backbone | COMPLETE — first valid DV benchmark |
| Q22 | Approximate parameter-matched tiny classical control | COMPLETE |
| Q23 | VinDr DV binary comparative report | COMPLETE — DV binary phase closed |
| Q24 | Roadmap realignment for CV binary phase | COMPLETE |
| Q25 | CV binary feasibility design | COMPLETE |
| Q25A | Roadmap prioritization and automation planning | COMPLETE |
| Q26 | CV binary pipeline smoke test (14/14 PASS) | COMPLETE |
| Q27 | CV binary full training (15 epochs, AUROC 0.6708) | COMPLETE |
| Q27A | NAS strategy and optimization phase refinement | COMPLETE |
| Q28 | DV vs CV binary comparative report | COMPLETE |
| Q29 | Binary quantum release tagging | IN PROGRESS (this slice) |

---

## 3. Final Frozen Benchmark Table

**This table is now frozen. All future comparative work must reference these values.**

The values below are drawn from the source reports. They are fixed reference points as of this release.

| Model | Slice | AUROC | F1 | Accuracy | Params |
|---|---|---|---|---|---|
| Classical CNN | Q17 | 0.6224 | 0.5355 | 60.66% | 23,650 |
| DV Hybrid | Q21 | 0.6800 | 0.6159 | 63.84% | 574 |
| Tiny Classical Control | Q22 | 0.6625 | 0.5961 | 64.37% | 526 |
| CV Hybrid | Q27 | 0.6708 | 0.6283 | 65.77% | 536 |

**Extended metric table (frozen):**

| Model | AUROC | F1 | Accuracy | Precision | Recall | AUPRC | Params |
|---|---|---|---|---|---|---|---|
| Q17 Classical CNN | 0.6224 | 0.5355 | 60.66% | 0.6263 | 0.4677 | 0.6730 | 23,650 |
| Q21 DV Hybrid | 0.6800 | 0.6159 | 63.84% | 0.6350 | 0.5978 | 0.6571 | 574 |
| Q22 Tiny Classical Control | 0.6625 | 0.5961 | 64.37% | 0.6618 | 0.5422 | 0.6559 | 526 |
| Q27 CV Hybrid | 0.6708 | 0.6283 | 65.77% | 0.6634 | 0.5968 | 0.6560 | 536 |

**Dataset and protocol (frozen):**

| Field | Value |
|---|---|
| Dataset | `data/processed/vindr_binary_roi_224` |
| Task | Binary: Any Pathology (1) vs No Finding (0) |
| Test set | 2,077 samples (1,070 class 0, 1,007 class 1) |
| Seed | 42 (single seed, all models) |
| Backbone (Q21/Q22/Q27) | `checkpoints/c006_d040_classical_anchor.pt` (C006-D040, PneumoniaMNIST-pretrained, Q6) |

---

## 4. Artifact Inventory

### Reports (frozen — do not modify)

| Report | Slice | Description | Status |
|---|---|---|---|
| `reports/vindr_classical_baseline_full_training.md` | Q17 | Classical CNN baseline full training | Frozen |
| `reports/vindr_dv_hybrid_pretrained_full_training.md` | Q21 | DV hybrid pretrained backbone full training | Frozen |
| `reports/vindr_classical_control_tiny_head.md` | Q22 | Tiny classical control training | Frozen |
| `reports/vindr_binary_comparative_report.md` | Q23 | VinDr DV binary comparative report | Frozen |
| `reports/q26_cv_binary_smoke_test.md` | Q26 | CV binary pipeline smoke test | Frozen |
| `reports/q27_cv_binary_full_training.md` | Q27 | CV binary full training | Frozen |
| `reports/q27a_nas_strategy_and_optimization_refinement.md` | Q27A | NAS strategy and optimization refinement | Frozen |
| `reports/q28_dv_vs_cv_binary_comparative_report.md` | Q28 | DV vs CV binary comparative report | Frozen |

### Scripts (frozen reference state)

| Script | Purpose | Status |
|---|---|---|
| `scripts/train_vindr_dv_hybrid_pretrained.py` | Q21 DV hybrid training | Frozen reference |
| `scripts/train_vindr_classical_control_tiny_head.py` | Q22 classical control training | Frozen reference |
| `scripts/smoke_test_vindr_cv_binary.py` | Q26 CV smoke validation | Frozen reference |
| `scripts/train_vindr_cv_binary.py` | Q27 CV full training | Frozen reference |

These scripts are frozen in the sense that they define the exact training procedures that produced the benchmarked results. Future scripts must not modify these files to retroactively alter the benchmark results. New scripts for Q30+ work are separate artifacts.

### Checkpoints (local only — not committed to repository)

| Checkpoint | Used By | Role |
|---|---|---|
| `checkpoints/c006_d040_classical_anchor.pt` | Q21, Q22, Q26, Q27 | Frozen pretrained backbone; source: PneumoniaMNIST training (Slice Q6); do not modify or overwrite |
| `checkpoints/vindr_dv_hybrid_pretrained_best.pt` | Q21 | Best DV hybrid model at epoch 15; frozen reference for Q21 benchmark |
| `checkpoints/vindr_classical_control_tiny_head_best.pt` | Q22 | Best tiny classical model at epoch 15; frozen reference for Q22 benchmark |
| `checkpoints/vindr_cv_binary_best.pt` | Q27 | Best CV hybrid model at epoch 15; frozen reference for Q27 benchmark |

**State:** No checkpoint file is committed to the repository (`.gitignore` covers `*.pt`). All checkpoint paths are local only. The benchmark values in the frozen table above are the authoritative record; checkpoints are supplementary artifacts for potential inference or reproducibility verification.

### Roadmap state at closure

| Slice / Phase | Status |
|---|---|
| Q17–Q28 | COMPLETE |
| Q29 | IN PROGRESS → COMPLETE (this slice) |
| Q30 | NEXT |
| Q31–Q35 | PLANNED — blocked until Q29 complete |
| Multiclass phase | BLOCKED — blocked until Q29 complete |
| NAS / AWS / Ray | BLOCKED — blocked until Q29 complete |

---

## 5. Git Release Tags

Three annotated tags have been created on commit `dc7fff6` (Q28 — DV vs CV binary comparative report). All tags are local. No tags have been pushed to any remote.

| Tag | Represents | Meaning |
|---|---|---|
| `qstrata-vindr-dv-binary-v1` | Q17, Q21, Q22, Q23 | VinDr DV binary benchmarking package finalized: classical baseline, DV hybrid, parameter-matched control, DV comparative report |
| `qstrata-vindr-cv-binary-v1` | Q25, Q26, Q27 | VinDr CV binary benchmarking package finalized: CV design, CV smoke test, CV full training |
| `qstrata-vindr-binary-comparative-v1` | Q28, full binary closure | First stable DV/CV comparative benchmark release: formal side-by-side comparison of all four model types |

**Tag creation commands executed:**

```bash
git tag -a qstrata-vindr-dv-binary-v1 -m "QStrata DV binary benchmarking release v1"
git tag -a qstrata-vindr-cv-binary-v1 -m "QStrata CV binary benchmarking release v1"
git tag -a qstrata-vindr-binary-comparative-v1 -m "QStrata binary comparative benchmarking release v1"
```

All three tags confirmed via `git tag --list`. No tags pushed.

---

## 6. Binary Benchmark Closure Logic

The binary benchmarking phase is now formally frozen. The results documented in Section 3 are fixed reference points as of this commit. The following constraints apply to all future work:

**Immutability.** The frozen benchmark table may not be retroactively altered. If a future experiment finds an error in a prior report, the correction is documented in a new report that references the original; the original frozen report is not overwritten. The frozen values define what was measured, not what was ideal.

**Reference requirement.** Any future experiment that claims improvement over the binary benchmarks must explicitly reference the frozen values from Section 3 and compute deltas against them. Comparisons against undocumented or informally remembered values are not valid.

**Scope.** The frozen benchmarks cover VinDr-SpineXR binary classification only, under the experimental conditions documented in Section 3 (seed 42, frozen C006-D040 backbone for Q21/Q22/Q27, same train/val/test split). Results do not generalize to other datasets, splits, backbone configurations, or tasks without explicit re-evaluation.

**Tag semantics.** The three annotated tags mark the commit at which each phase was declared complete. They are not version numbers implying future `v2` releases — they are historical markers. If a future phase (e.g., post-NAS benchmarking) produces a new reference table, it will carry its own tag with its own scope.

---

## 7. Scientific Integrity Discussion

**The Q22 control was necessary, not optional.** The Q21 DV hybrid outperformed Q17 on AUROC (+0.0576) and F1 (+0.0804). Without Q22, this improvement could have been entirely attributed to the compact bottleneck effect or the frozen pretrained backbone, neither of which is a quantum property. Q22 showed that a classical head of approximately equivalent parameter count on the same frozen backbone recovers 70–75% of that gain — meaning the compact bottleneck plus frozen backbone is the dominant driver. Q21 retains a residual AUROC advantage (+0.0175 over Q22) that the bottleneck effect does not fully explain. This residual is scientifically interesting, but it is a single-seed point estimate without statistical validation. The Q22 control is what makes the Q21 result interpretable.

**The compactness result is real but attributed correctly.** Q21, Q22, and Q27 all use ~500–580 trainable parameters against Q17's 23,650 and all outperform Q17 on AUROC and F1. This compactness result is not primarily explained by the quantum heads — the frozen pretrained backbone contributes 9,612 parameters of fixed capacity, and the compact head provides strong regularization regardless of type. The improvement is attributed to the architectural pattern (frozen pretrained backbone + compact trainable head) rather than to any quantum property.

**The AUROC–F1 inversion is noted, not interpreted.** DV leads AUROC (+0.0092 over CV), CV leads F1 (+0.0124 over DV). This inversion is a genuine single-seed observation documented in Q28. It is noted as a potential direction for future investigation — different head architectures may have different threshold-sensitivity profiles — but it is not interpreted as evidence of DV superiority, CV superiority, or architectural differentiation. Single-seed inversions at small magnitude are uninformative without multi-seed confirmation.

**Framing integrity.** The benchmarking sequence was framed from the start as exploratory compact research under a cross-domain pretrained backbone with a single seed. This framing is preserved in all closure documentation. No scope creep occurred: the results are described as what they are, not as what would be desirable to claim.

**No quantum advantage claim.** At no point in this benchmarking phase — not in Q21, Q22, Q23, Q27, Q28, or this report — is quantum advantage claimed, implied, or suggested. The DV and CV quantum hybrid results are described as exploratory benchmarks with small residual advantages over a parameter-matched classical control. The appropriate conclusion is that the quantum hybrid architecture warrants further investigation under stronger experimental controls, not that it has demonstrated superiority.

---

## 8. Reproducibility and Research Discipline

**Annotated release tags as historical anchors.** The three annotated tags (`qstrata-vindr-dv-binary-v1`, `qstrata-vindr-cv-binary-v1`, `qstrata-vindr-binary-comparative-v1`) mark the exact git commit at which the binary benchmarking phase was declared complete. Any future researcher or collaborator can recover the precise project state at binary closure by:

```bash
git checkout qstrata-vindr-binary-comparative-v1
```

This will restore the exact versions of all scripts, reports, roadmap, and `qcore` files as they existed at binary closure. The benchmark results are reproducible from this state, subject to checkpoint availability.

**Frozen reports as sufficient scientific records.** The frozen reports (Section 4) contain exact per-epoch metrics, confusion matrices, gradient health logs (for Q21), and CV health metrics (for Q27). These records are sufficient to reconstruct the scientific interpretation of the benchmarking phase without re-running any experiment. The interpretation in Q28 was derived from these records only; no value was filled from memory.

**Gating discipline enforced.** The roadmap gating structure — NAS blocked until Q29, multiclass blocked until Q29 — enforced sequential validation rather than premature scaling. This discipline is documented here as part of the reproducibility record. The decision to gate automation and search work until the scientific baseline was formally closed was made at Q25A and upheld through Q26, Q27, Q27A, Q28, and this slice. Future work inherits this precedent: scientific validation before automation.

**Single-seed caveat is explicit and forward-looking.** All four benchmark models used seed 42 and a single training run. This is documented in every relevant report. It is a known limitation that has been explicitly carried forward into the limitations sections of Q23, Q28, and this report. Multi-seed evaluation is identified as the highest-priority next step for any future work that seeks to draw architectural conclusions from the observed deltas.

---

## 9. Transition to Optimization Phase

With Q29 complete, the Q30–Q35 optimization and NAS phase is unblocked. The transition proceeds as follows:

**Q30 — Experiment Automation Framework Design.** Designs a reproducible local experiment orchestration framework covering: metric tracking schema, checkpoint management conventions, sweep configuration format (YAML), artifact naming, and logging structure. Output is a design document. No code is written in Q30. The design must be reviewed and approved before Q31 begins.

**Q31 — Local GPU Experiment Runner.** Implements the framework designed in Q30. Produces a working local runner with documented examples. Supports: run queuing, resume of failed runs, structured metric collection, result ranking by configurable objectives, and reproducible sweep logs.

**Q32 — NAS Search Space Design — Classical Feature Extractors.** Defines and searches a constrained space of compact classical CNN architectures: convolution block types, channel widths, depth variants, pooling strategies. The goal is to establish the strongest compact classical architecture achievable on VinDr-SpineXR binary under the same parameter budget. This is the **classical ceiling experiment**. It must run before quantum NAS.

**Q33 — NAS Search Space Design — Quantum Heads (DV/CV).** Defines constrained search spaces for DV heads (qubits, circuit depth, ansatz type) and CV heads (modes, circuit depth, squeezing range, encoding strategy). Q33 begins only after Q32 has produced a validated classical ceiling. The Q32 classical ceiling defines the evaluation criteria and stopping conditions for Q33.

**Q34 — Local Multi-Objective NAS Pilot.** Executes joint optimization across AUROC, F1, parameter count, inference latency, and training stability on a single GPU. Multi-objective, not single-metric. No AWS or Ray. Produces a Pareto frontier; final model selection is a scientific decision.

**Q35 — AWS / Ray Distributed Scaling Design.** Designs distributed NAS only after Q34 local NAS is validated. Covers AWS GPU provisioning, Ray orchestration, artifact management, cost controls. Output is a design document. No cloud infrastructure is provisioned before design approval.

**Classical ceiling principle (carried forward).** Classical NAS (Q32) must always precede quantum NAS (Q33). The frozen benchmark table in Section 3 defines the current reference floor (Q17 AUROC 0.6224, Q22 AUROC 0.6625). Q32 will define the ceiling. Until the ceiling is known, no quantum NAS result can be scientifically interpreted — it is impossible to determine whether a found quantum architecture beats the best attainable classical architecture at the same parameter budget.

---

## 10. Required Scientific Guardrail

> Q29 formally closes the exploratory VinDr-SpineXR binary quantum benchmarking phase under the QStrata framework. The released benchmarks are compact exploratory research baselines and do NOT establish quantum advantage, statistical superiority, or clinical readiness.

---

## 11. Closure Status

| Phase | Status |
|---|---|
| VinDr-SpineXR binary benchmarking (Q17–Q29) | **CLOSED** |
| NAS phase (Q30–Q35) | NEXT — unblocked by Q29; Q30 is immediate next slice |
| Multiclass phase | BLOCKED — pending future roadmap progression after Q29; multiclass slices may now be scheduled |
| PneumoniaMNIST comparative report (P21) | TODO — not yet scheduled |
| Global binary benchmark summary (R-FINAL) | TODO — deferred until PneumoniaMNIST comparative (P21) complete |

The binary benchmarking phase covers VinDr-SpineXR only. PneumoniaMNIST binary comparative work (P21) and the global summary (R-FINAL) remain on the TODO list and are not blocked, but are also not the immediate execution priority. Q30 is the immediate next slice.

---

## 12. Next Slice

**Q30 — Experiment Automation Framework Design**

Purpose: design a reproducible local experiment orchestration framework as the foundation for all Q31–Q35 NAS and optimization work. Output is a design document covering metric tracking, checkpoint management, sweep configuration, artifact naming, and logging structure. No training runs, no model changes, no code committed in Q30.

Gate: Q30 begins immediately — Q29 (this slice) is its prerequisite and is now complete.

---

```
Q29 status: COMPLETE
Binary benchmarking phase: CLOSED
Git tags created: qstrata-vindr-dv-binary-v1, qstrata-vindr-cv-binary-v1, qstrata-vindr-binary-comparative-v1
Q30 status: NEXT — Experiment Automation Framework Design
NAS/AWS/Ray: unblocked (Q30 is gate)
Multiclass: unblocked (scheduling pending)
```
