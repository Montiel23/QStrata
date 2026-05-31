# Q46A — Feature Extractor Benchmark Plan

**Date:** 2026-05-31
**Slice:** Q46A
**Branch:** feature/q46a_feature_extractor_benchmark_plan
**Type:** Planning only — no training, no benchmark execution
**Blocked until:** Q45B (partial fine-tuning benchmark) complete

---

## Protocol Summary

| Element | Value |
|---|---|
| Seeds | 42, 7, 123, 999, 2025 |
| Runtime cap | 240 min total (Phase 1: 60 min, Phase 2: 120 min, Phase 3: 60 min) |
| Decision rule | Winner must have mean AUROC > Q45A baseline (0.7196); negative if no candidate clears |
| Stop conditions | All Phase 1 candidates below baseline − 0.030; OOM/crash excluded; wall-time cap enforced |
| Expected output artifacts | `experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv`, `experiments/leaderboards/q46b_extractor_full_leaderboard.csv`, `reports/q46b_feature_extractor_benchmark.md` |

---

## 1. Objective

Define a bounded, execution-ready protocol for benchmarking compact feature extractor (backbone) alternatives against the current frozen baseline on VinDr-SpineXR binary classification.

The current production pipeline uses the pretrained C006-D040 backbone (frozen ResNet feature extractor) established in Phase 1. All binary uplift work through Q45A has been conducted against this frozen backbone. Phase 6b explicitly lists "Extractor / Backbone" as an optimization dimension (compact backbone replacement: EfficientNet-B0, MobileNetV3, ConvNeXt-Tiny).

Q46A produces the execution protocol only. Execution is deferred to Q46B.

---

## 2. Current Binary Ceiling

| Metric | Value | Config | Source |
|---|---|---|---|
| AUROC | 0.7239 | CLAHE clip=3.0 tile=4×4, frozen C006-D040, no augmentation | Q38C |
| F1 | 0.6779 | same | Q38C |
| Mean AUROC (5-seed) | 0.7168 ± 0.0096 | CLAHE-only, 5 seeds | Q43 |
| Mean AUROC (3-seed) | 0.7196 ± 0.0095 | CLAHE-only, 3 seeds | Q45A |

All gains through Q45A used the same frozen backbone. The ceiling reflects a well-tuned preprocessing pipeline but an unexplored extractor dimension.

---

## 3. Candidates

Five feature extractors are benchmarked in Q46B. All use ImageNet-pretrained weights (frozen in Q46B Phase 1; unfreezing explored in Q46C if Q46B warrants it).

| ID | Backbone | Params (extractor) | Notes |
|---|---|---|---|
| `baseline` | C006-D040 frozen ResNet (current) | ~11M frozen | Reference; all prior Phase 6b results |
| `efficientnet_b0` | EfficientNet-B0 (torchvision) | ~4.0M frozen | Compact; strong ImageNet accuracy-per-param ratio |
| `mobilenetv3_small` | MobileNetV3-Small (torchvision) | ~0.93M frozen | Ultra-compact; lowest params |
| `mobilenetv3_large` | MobileNetV3-Large (torchvision) | ~3.0M frozen | Mid-range compact; better accuracy than Small |
| `convnext_tiny` | ConvNeXt-Tiny (torchvision) | ~27M frozen | Modern architecture; higher param count but available pretrained |

**Extraction interface:** All candidates expose a fixed-dimension feature vector fed to the same q34a_trial_004 classification head (2,250 head params). Feature projection layer (linear) added per backbone as needed to match the head input dimension.

**Why these five:** Covers the compact-backbone trifecta named in the Phase 6b roadmap (EfficientNet-B0, MobileNetV3, ConvNeXt-Tiny) plus the current baseline and a MobileNetV3-Large variant for gradient coverage between Small and Large.

---

## 4. Evaluation Protocol

### 4.1 Fixed experimental parameters

| Parameter | Value | Source |
|---|---|---|
| Preprocessing | CLAHE clip=3.0 tile=4×4 | Q38C champion |
| Augmentation | None | Q45A closure decision |
| Epochs | 4 | Consistent with Q38A–Q45A |
| Learning rate | 1e-03 | Q38A–Q45A standard |
| Weight decay | 1e-04 | Q38A–Q45A standard |
| Loss | BCEWithLogitsLoss | Phase 1 standard |
| Optimizer | Adam | Phase 1 standard |
| DataLoader | bs=8 nw=4 pm=True pw=True pf=2 | Q41 optimised profile |

### 4.2 Seeds

| Phase | Seeds | Purpose |
|---|---|---|
| Smoke (Phase 1) | `[42]` | Single-seed viability filter |
| Full evaluation (Phase 2) | `[42, 7, 123]` | 3-seed multi-seed evaluation |
| Extended (Phase 3, if triggered) | `[42, 7, 123, 999, 2025]` | 5-seed rigorous validation — only if Phase 2 winner is within 0.005 AUROC of ceiling |

**Seeds:** 42, 7, 123, 999, 2025

### 4.3 Metrics collected per seed per candidate

- AUROC (primary)
- F1 (secondary)
- Accuracy
- Params (head only; backbone always frozen)
- Wall time per epoch (seconds)
- Preprocessing overhead (ms/image, measured once per backbone)

### 4.4 Aggregation

- Mean ± std over seeds
- 95% CI via t-distribution (df = n\_seeds − 1)
- ΔAUROC vs Q45A baseline (mean AUROC 0.7196)
- ΔAUROC vs Q38C ceiling (0.7239)

---

## 5. Runtime Cap

| Phase | Scope | Cap |
|---|---|---|
| Smoke (Phase 1) | 1 seed × 5 candidates × 4 epochs | 60 minutes wall time |
| Full evaluation (Phase 2) | 3 seeds × top-K candidates × 4 epochs | 120 minutes wall time |
| Extended (Phase 3) | 5 seeds × winner × 4 epochs | 60 minutes wall time |

**Total runtime cap (all phases combined):** 240 minutes wall time.

If Phase 1 smoke exceeds 60 min, terminate, diagnose the outlier backbone, and reduce candidates before proceeding.

---

## 6. Decision Rule

### Phase 1 → Phase 2 gate

Advance a candidate from smoke to full evaluation if:

```
smoke_auroc(seed=42) > baseline_mean_auroc(Q45A) - 0.010
```

i.e., within 1.0pp below the Q45A 3-seed mean (0.7196). Candidates more than 1.0pp below the baseline are pruned before Phase 2.

### Winner selection (Phase 2 output)

The Phase 2 winner is the candidate with the highest mean AUROC across 3 seeds, subject to:

1. `mean_auroc > Q45A_3seed_mean (0.7196)` — must beat the current multi-seed baseline.
2. `95% CI lower bound > Q38C_best (0.7239) - 0.020` — must be statistically plausible to reach or exceed the Q38C single-seed ceiling.
3. If multiple candidates satisfy (1) and (2): prefer the candidate with fewer extractor parameters (compact first).

If no candidate satisfies condition (1): **extractor phase result is NEGATIVE** — document and proceed to Q46 (CV head re-evaluation) with the current C006-D040 baseline.

### Phase 3 trigger

Trigger 5-seed extended validation if:

```
winner_mean_auroc (Phase 2) ∈ [0.7239 − 0.005, 0.7239 + 0.005]
```

i.e., winner is within 0.5pp of the Q38C ceiling in either direction (may have exceeded it or may just be noise).

---

## 7. Stop Conditions

| Condition | Action |
|---|---|
| All Phase 1 candidates below `baseline_mean − 0.030` | Stop; record result as NEGATIVE; skip Phase 2; proceed to Q46 |
| Phase 1 runtime exceeds 60 min | Pause; diagnose slowest backbone; prune or reduce epochs; resume |
| Phase 2 runtime exceeds 120 min | Terminate; use completed seeds only if ≥ 2 seeds per candidate; else pause and resume |
| Any backbone causes OOM or crash in Phase 1 | Exclude candidate; document; continue Phase 1 with remaining candidates |
| Winner Phase 2 mean AUROC > 0.78 (above near-term ceiling) | Flag as unexpected uplift; trigger Phase 3 unconditionally |

---

## 8. Expected Output Artifacts

### Q46B execution outputs (future)

| Artifact | Path | Description |
|---|---|---|
| Smoke leaderboard | `experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv` | Phase 1 results, 1 seed |
| Full leaderboard | `experiments/leaderboards/q46b_extractor_full_leaderboard.csv` | Phase 2 results, 3 seeds |
| Benchmark report | `reports/q46b_feature_extractor_benchmark.md` | Full Q46B report with decision, CI table, recommendation |
| Updated roadmap | `docs/roadmaps/qstrata_master_research_roadmap.md` | Q46B status updated to COMPLETE or NEGATIVE |

### Q46C expected outputs (if Q46B winner identified)

| Artifact | Path | Description |
|---|---|---|
| CV head leaderboard | `experiments/leaderboards/q46c_cv_head_extractor_leaderboard.csv` | q34c_trial_005 head on Q46B winner backbone |
| CV head report | `reports/q46c_cv_head_re_evaluation.md` | AUROC/F1 of Q35 canonical CV candidate on new extractor |

### Q47 expected outputs (binary uplift report)

| Artifact | Path | Description |
|---|---|---|
| Binary uplift report | `reports/q47_binary_uplift_comparative_report.md` | End-to-end Phase 6b comparative: preprocessing → augmentation → extractor → fine-tuning → CV head |

---

## 9. Dependency Chain

```
Q45A COMPLETE (augmentation CLOSED, clahe_no_augmentation confirmed)
  └─► Q45B (partial fine-tuning benchmark) — NEXT
        └─► Q46A (this plan) — PLANNING COMPLETE
              └─► Q46B (feature extractor benchmark execution) — BLOCKED on Q45B ✓
                    └─► Q46C (CV head re-evaluation on Q46B winner backbone)
                          └─► Q47 (binary uplift comparative report)
                                └─► Phase 7 (multiclass) UNBLOCKED
```

**Q46B is blocked until Q45B is complete.** Partial fine-tuning results may reveal which backbone layers are most plastically responsive on this dataset, informing whether frozen-only evaluation in Q46B is sufficient or whether a follow-up partially-finetuned extractor benchmark (Q46D) is warranted.

---

## 10. Reproducibility Requirements

- All Q46B runs must record: git commit SHA, Python version, torch version, torchvision version, seed, hostname, GPU/CPU model.
- Each backbone candidate must use the same fixed random seed initialization for the projection + classification head (i.e., backbone weights from torchvision ImageNet pretrained; head from q34a_trial_004 architecture, re-initialized with `seed` for each trial).
- DataLoader profile pinned to Q41 optimised settings (no deviations without documentation).
- Results reproducible from frozen config + git SHA.

---

## 11. Scientific Constraints

- **No multiclass until Phase 6b complete.** Q46B results feed the binary uplift comparative report (Q47); multiclass (Phase 7) remains gated on Phase 6b closure.
- **Anti-overengineering:** Do not implement custom backbone architectures, backbone NAS, or combined ablations until individual extractor effects are quantified.
- **Frozen-first:** All Q46B Phase 1 and Phase 2 runs use fully frozen backbone weights. Partial unfreezing is reserved for Q46C/Q46D if individual extractor gains warrant it.
- **One dimension at a time:** Q46B benchmarks extractor choice in isolation. CLAHE preprocessing and no-augmentation are fixed inputs from prior Phase 6b decisions; they are not re-swept.

---

## 12. Validation Checklist (for Q46B execution)

Before executing Q46B, verify:

- [ ] Q45B complete and best fine-tuning config documented
- [ ] All five backbone candidates importable via `torchvision.models` with `weights="DEFAULT"`
- [ ] Projection layer dimension confirmed per backbone (`EfficientNet-B0: 1280`, `MobileNetV3-Small: 576`, `MobileNetV3-Large: 960`, `ConvNeXt-Tiny: 768`, `baseline: existing`)
- [ ] Q41 DataLoader profile confirmed in run script CLI defaults
- [ ] Seeds `[42, 7, 123, 999, 2025]` defined as constants in run script
- [ ] Runtime cap enforcement in run script (wall-time check between candidates)
- [ ] Leaderboard CSV schema matches Phase 2 aggregation format
