# QStrata

**Medical imaging model optimization R&D** — systematic evaluation of classical and quantum hybrid deep learning architectures under compact parameter budgets.

---

## Project Status

**Current phase:** Phase 6b — Binary Performance Uplift (NEXT: Q39 Augmentation Benchmark)  
**Roadmap:** [`docs/roadmaps/qstrata_master_research_roadmap.md`](docs/roadmaps/qstrata_master_research_roadmap.md)

### Canonical Compact Candidates (Q35 Cross-Frontier Pareto + Q38A CLAHE)

| Model | AUROC | F1 | Params | Source |
|---|---|---|---|---|
| Classical + CLAHE | **0.6962** | 0.6201 | 2,250 | q34a_trial_004 + Q38A |
| Classical compact | 0.6835 | 0.6398 | 2,250 | q34a_trial_004 |
| CV Quantum best F1 | 0.6623 | **0.6463** | **274** | q34c_trial_005 |

DV quantum: entirely dominated by CV on all four objectives — excluded from further NAS.  
Q38A finding: CLAHE is the only preprocessing that improves AUROC (+1.27pp); all normalization methods degrade performance.

---

## Research Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Binary Benchmarking | **CLOSED** |
| 2 | Experiment Automation | **COMPLETE** |
| 3 | Classical NAS Ceiling | IN PROGRESS |
| 4 | Quantum NAS | IN PROGRESS |
| 5 | Local NAS Pilot | **COMPLETE** |
| 5b | Unified Pareto Analysis | **COMPLETE** |
| 6 | Cloud Validation (SkyPilot) | IN PROGRESS |
| **6b** | **Binary Performance Uplift** | **NEXT** |
| 7 | Multiclass Benchmarking | BLOCKED (Phase 6b gate) |

---

## Hard Constraints

- **No Ray, no distributed NAS** — cloud validation is single-node only (SkyPilot)
- **No multiclass until Phase 6b complete** — binary uplift must stabilize baselines first
- **No object detection** — out of current scope
- **Standard library first** — no speculative external dependencies
- **Local-first** — all experiments reproducible locally before cloud execution

---

## Key Artifacts

| Artifact | Path |
|---|---|
| Master roadmap | `docs/roadmaps/qstrata_master_research_roadmap.md` |
| Q38A preprocessing report | `reports/q38a_binary_preprocessing_benchmark.md` |
| Q38A preprocessing leaderboard | `experiments/leaderboards/q38a_preprocessing_leaderboard.csv` |
| Q35 unified frontier | `experiments/leaderboards/q35_unified_frontier.csv` |
| Q35 analysis report | `reports/q35_unified_pareto_frontier_analysis.md` |
| SkyPilot smoke YAML | `infra/skypilot/q36a_single_node_smoke.yaml` |
| CV NAS pilot report | `reports/q34c_cv_nas_pilot_mvp.md` |
| Classical NAS report | `reports/q34a_classical_nas_pilot_mvp.md` |

---

## Infrastructure

- **Local:** Docker GPU (`docker-qstrata-gpu-1`, RTX 2060 Super, 12GB)
- **Cloud:** SkyPilot + AWS (c6i.xlarge, CPU-only, $0.17/hr) — Q36B PARTIAL
- **Dataset:** VinDr-SpineXR binary ROI (local); PneumoniaMNIST (auto-download via medmnist)
