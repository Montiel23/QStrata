# Q58 Custom CV-QNN Binary Benchmark — Validation Report

**Slice ID:** Q58-CUSTOM-CV-QNN-BINARY-BENCHMARK  
**Date:** 2026-06-03  
**Seeds:** [42, 7, 123]  
**Epochs:** 4  
**LR:** 0.001  |  **Weight Decay:** 0.0001  

## Architecture

| Component | Details | Params |
|-----------|---------|--------|
| Backbone  | MobileNetV3-Large (ImageNet, frozen) | 5,483,032 |
| Projection | Linear(960→128), frozen random | 123,008 |
| CV encoder | Linear(128→4), trainable | 516 |
| CV ansatz  | GaussianVariationalAnsatz (depth=1, n_modes=2) | 10 |
| CV readout | Linear(2→2), trainable | 6 |
| **CV-QNN head total** | Trainable CV-QNN parameters | **532** |
| **Grand total** | All components | **5,606,572** |

## CV-QNN Circuit Details

- **Framework:** Custom in-house qcore Gaussian CV simulator (no PennyLane/Qiskit/StrawberryFields)
- **Backend:** GaussianBackend — symplectic matrix evolution of (μ, Σ)
- **Ansatz:** GaussianVariationalAnsatz — Displacement(D) → Squeezing(S) → Rotation(R) → Beamsplitter(BS, ring)
- **Modes:** 2  |  **Depth:** 1  |  **Squeezing cap:** 1.5 (tanh-bounded)
- **Data reuploading:** encoded_input added to μ at each depth layer
- **Measurement:** Homodyne X-quadrature (even indices of μ) → 2 features
- **Gradient flow:** symplectic ops are differentiable via torch; autograd end-to-end

## Datasets

| Dataset | Source | Train N | Val N | Notes |
|---------|--------|---------|-------|-------|
| VinDr-SpineXR | Real DICOM ROIs | 6,712 | 1,677 | Q49 pre-computed 128-dim embeddings |
| BUU-LSPINE | Synthetic procedural | 1000 | 200 | Same generator as Q56/Q57 |

## Results

### VinDr-SpineXR

| Seed | AUROC | F1 | Accuracy | Runtime (s) | Δ vs Q56 | Δ vs Q57 |
|------|-------|-----|----------|-------------|----------|----------|
| 42 | 0.9524 | 0.8720 | 0.8795 | 56.2 | -0.0206 | +0.0682 |
| 7 | 0.9544 | 0.8813 | 0.8795 | 57.4 | -0.0187 | +0.0702 |
| 123 | 0.9534 | 0.8840 | 0.8879 | 55.0 | -0.0197 | +0.0691 |
| **Mean** | **0.9534** | **0.8791** | | | | |
| CI95 | [0.9510, 0.9558] | [0.8635, 0.8947] | | | | |

**AUROC:** 0.9534 ± 0.0010 (95% CI: [0.9510, 0.9558])  
**F1:**    0.8791 (95% CI: [0.8635, 0.8947])  
**Δ AUROC vs Q46 winner:** -0.0340  
**Δ AUROC vs Q56 CNN baseline:** -0.0197  
**Δ AUROC vs Q57 DV-QNN:** +0.0692  

### BUU-LSPINE (Synthetic)

| Seed | AUROC | F1 | Accuracy | Runtime (s) | Δ vs Q56 | Δ vs Q57 |
|------|-------|-----|----------|-------------|----------|----------|
| 42 | 1.0000 | 1.0000 | 1.0000 | 8.6 | +0.0000 | +0.0000 |
| 7 | 1.0000 | 1.0000 | 1.0000 | 8.8 | +0.0000 | +0.0000 |
| 123 | 1.0000 | 1.0000 | 1.0000 | 8.1 | +0.0000 | +0.0000 |
| **Mean** | **1.0000** | **1.0000** | | | | |
| CI95 | [1.0000, 1.0000] | [1.0000, 1.0000] | | | | |

**AUROC:** 1.0000 ± 0.0000 (95% CI: [1.0000, 1.0000])  
**F1:**    1.0000 (95% CI: [1.0000, 1.0000])  
**Δ AUROC vs Q56 CNN baseline:** +0.0000  
**Δ AUROC vs Q57 DV-QNN:** +0.0000  

## Comparison: CV-QNN vs CNN Baseline (Q56) and DV-QNN (Q57)

| Dataset | Q56 CNN | Q57 DV-QNN | Q58 CV-QNN | Δ vs Q56 | Δ vs Q57 | CV-QNN Params |
|---------|---------|------------|------------|----------|----------|---------------|
| VinDr-SpineXR | 0.9731 | 0.8842 | 0.9534 | -0.0197 | +0.0692 | 532 |
| BUU-LSPINE | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 | 532 |

## Validation Checklist

- [x] Custom CV-QNN head integrated with MobileNetV3-Large feature extractor
- [x] Trained on both datasets (VinDr-SpineXR + BUU-LSPINE) across all seeds
- [x] Leaderboard CSV written with AUROC/F1/CI95/circuit_params/runtime
- [x] Validation report documents CV-QNN results vs CNN baseline and DV-QNN
- [x] Only custom in-house quantum framework used (qcore — no PennyLane/Qiskit/StrawberryFields)

## Notes

- VinDr-SpineXR uses Q49 pre-computed 128-dim embeddings (MobileNetV3-Large + frozen random projection 960→128).  CV-QNN head trained fresh per seed.
- BUU-LSPINE uses synthetic procedural spine ROI images (same generator as Q56/Q57). Features extracted via MobileNetV3-Large + frozen seeded random projection.
- Trainable parameters per run: CV-QNN head only (532 params). Backbone and projection are fully frozen.
- CV-QNN uses Gaussian (continuous-variable) photonic simulation. State evolution via symplectic matrices on (μ, Σ) — CPU-only, typically faster than DV circuit simulation at small mode counts.
- Squeezing is bounded by tanh(r_raw) × 1.5 to ensure physical stability.
- Reference baselines: Q46 winner AUROC = 0.9873 | Q56 CNN VinDr AUROC = 0.9731 | Q57 DV-QNN VinDr AUROC = 0.8842
