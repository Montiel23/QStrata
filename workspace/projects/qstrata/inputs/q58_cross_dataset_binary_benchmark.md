# Q58 — Cross-Dataset Binary Benchmark

**Slice ID**: Q58-CROSS-DATASET-BINARY-BENCHMARK  
**Campaign**: binary_classical_publication_package  
**Status**: BLOCKED  
**Depends on**: Q57 (BLOCKED on Q56)  
**Estimated runtime**: MEDIUM (30–60 min)  
**Date planned**: After Q57 completes  

---

## 1. Objective

Produce the unified cross-dataset binary classification comparison table that will anchor the
QStrata publication. Combines:

1. VinDr-SpineXR binary results (Q17–Q49 + Q56–Q57 campaign winners)
2. PneumoniaMNIST binary baselines (frozen from prior work — no re-execution)

This is the broadest comparison in the campaign and directly supports the "classical vs quantum"
narrative in the paper.

---

## 2. Scope

### 2.1 VinDr-SpineXR Models (all results frozen — sourced from existing reports)

| Model | AUROC | F1 | Params | Source |
|---|---|---|---|---|
| Q17 Classical CNN (3-block) | 0.6224 | 0.5355 | 23,650 | `reports/vindr_classical_baseline_full_training.md` |
| Q21 DV Hybrid (pretrained backbone) | 0.6800 | 0.6159 | 574 | `reports/vindr_dv_hybrid_pretrained_full_training.md` |
| Q22 Tiny Classical Control | 0.6625 | 0.5961 | 526 | `reports/vindr_classical_control_tiny_head.md` |
| Q27 CV Hybrid | 0.6708 | 0.6283 | 536 | `reports/q27_cv_binary_full_training.md` |
| Q34A Compact Classical NAS (trial_004) | 0.6835 | 0.6398 | 2,250 | `reports/q34a_classical_nas_pilot_mvp.md` |
| Q34C CV NAS (trial_005) | 0.6623 | 0.6463 | 274 | `reports/q34c_cv_nas_pilot_mvp.md` |
| Q46B MobileNetV3-Large (AUROC 3-seed mean) | 0.9873 | 0.9331 | 2,974,202 | `reports/q48_feature_extractor_phase_closure.md` |
| Q56 Winner (discrete classifier) | TBD | TBD | TBD | Q56 |
| Q57 Winner (continuous classifier) | TBD | TBD | TBD | Q57 |

### 2.2 PneumoniaMNIST Models (frozen reference — no re-execution required)

| Model | AUROC | F1 | Params | Source |
|---|---|---|---|---|
| PneumoniaMNIST Classical Baseline (C006-D040) | (from reports) | (from reports) | 9,612 | `reports/c006_d040_checkpoint_training.md` |
| PneumoniaMNIST DV Hybrid (pretrained) | (from reports) | (from reports) | 574 | `reports/dv_hybrid_pneumoniamnist_pretrained_baseline.md` |

**Note:** PneumoniaMNIST values are pulled from existing frozen reports. No re-training is performed.

---

## 3. Protocol

### 3.1 Data Collection

1. Load Q17–Q46B VinDr-SpineXR results from existing reports (CSV/JSON artifacts)
2. Load Q56 and Q57 final leaderboard winners
3. Load PneumoniaMNIST baselines from frozen reports
4. Populate unified comparison table

### 3.2 Comparison Dimensions

For each model, record:

| Column | Notes |
|---|---|
| model_id | Unique identifier |
| dataset | VinDr-SpineXR or PneumoniaMNIST |
| model_family | classical_cnn / dv_hybrid / cv_hybrid / classical_ml_discrete / classical_ml_continuous / pretrained_backbone |
| test_auroc | Mean across seeds (where available) |
| test_auroc_std | Std across seeds (where available) |
| test_f1 | Mean across seeds |
| test_accuracy | Mean across seeds |
| trainable_params | Total trainable parameter count |
| backbone_frozen | true/false |
| preprocessing | CLAHE config or none |
| seeds_used | Comma-separated list |
| source_slice | Q-number |
| source_report | Path to report |

### 3.3 Cross-Dataset Comparison Rules

- AUROC is the primary cross-dataset comparison metric (threshold-independent)
- Do NOT claim cross-dataset generalization — each dataset is analyzed separately
- Unified table is for publication overview only — no inference about one dataset from the other
- Quantum vs classical comparisons use parameter-matched pairs where possible (Q21 vs Q22, Q34C vs Q34A)

---

## 4. Outputs

| File | Description |
|---|---|
| `workspace/experiments/Q58/leaderboards/q58_cross_dataset_leaderboard.csv` | Unified cross-dataset comparison table |
| `workspace/experiments/Q58/results/q58_cross_dataset_results.json` | Full results dict with provenance |
| `workspace/experiments/Q58/reports/q58_cross_dataset_benchmark_report.md` | Narrative analysis |

---

## 5. Pass Criteria

- [ ] Unified comparison table includes all listed VinDr + PneumoniaMNIST models
- [ ] All AUROC values traceable to named source reports
- [ ] No re-training performed — data collection only
- [ ] Cross-dataset comparison rules respected (no generalization claims)
- [ ] No source code modified; no git commit made
