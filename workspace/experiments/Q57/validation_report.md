# Q57 Custom DV-QNN Binary Benchmark — Validation Report

**Slice ID:** Q57-CUSTOM-DV-QNN-BINARY-BENCHMARK  
**Date:** 2026-06-03  
**Seeds:** [42, 7, 123]  
**Epochs:** 4  
**LR:** 0.001  |  **Weight Decay:** 0.0001  

## Architecture

| Component | Details | Params |
|-----------|---------|--------|
| Backbone  | MobileNetV3-Large (ImageNet, frozen) | 5,483,032 |
| Projection | Linear(960→128), frozen random | 123,008 |
| QNN proj   | Linear(128→4), trainable | 516 |
| QNN theta  | Variational angles (depth=1, n_qubits=4) | 24 |
| QNN readout| Linear(16→2), trainable | 34 |
| **QNN head total** | Trainable DV-QNN parameters | **574** |
| **Grand total** | All components | **5,606,614** |

## DV-QNN Circuit Details

- **Framework:** Custom in-house qcore DV simulator (no PennyLane/Qiskit/TorchQuantum)
- **Ansatz:** `medical_ansatz` — H init + data reuploading (atan) + variational + ring+cross CNOT
- **Qubits:** 4  |  **Depth:** 1  |  **Alpha:** 0.1
- **State space:** 16-dimensional Hilbert space
- **Circuit ops per sample:** 4H + 8 encoding + 12 variational + 8 CNOT = 32 gates
- **Gradient flow:** data reuploading via torch.atan preserves autograd end-to-end

## Datasets

| Dataset | Source | Train N | Val N | Notes |
|---------|--------|---------|-------|-------|
| VinDr-SpineXR | Real DICOM ROIs | 6,712 | 1,677 | Q49 pre-computed 128-dim embeddings |
| BUU-LSPINE | Synthetic procedural | 1000 | 200 | Same generator as Q56 |

## Results

### VinDr-SpineXR

| Seed | AUROC | F1 | Accuracy | Runtime (s) | Δ vs Q56 |
|------|-------|-----|----------|-------------|----------|
| 42 | 0.8833 | 0.7785 | 0.7984 | 263.8 | -0.0898 |
| 7 | 0.8844 | 0.7923 | 0.8014 | 261.7 | -0.0886 |
| 123 | 0.8849 | 0.7956 | 0.8020 | 265.1 | -0.0882 |
| **Mean** | **0.8842** | **0.7888** | | | |
| CI95 | [0.8822, 0.8863] | [0.7663, 0.8113] | | | |

**AUROC:** 0.8842 ± 0.0008 (95% CI: [0.8822, 0.8863])  
**F1:**    0.7888 (95% CI: [0.7663, 0.8113])  
**Δ AUROC vs Q46 winner:** -0.1031  
**Δ AUROC vs Q56 CNN baseline:** -0.0888  

### BUU-LSPINE (Synthetic)

| Seed | AUROC | F1 | Accuracy | Runtime (s) | Δ vs Q56 |
|------|-------|-----|----------|-------------|----------|
| 42 | 1.0000 | 1.0000 | 1.0000 | 39.4 | +0.0000 |
| 7 | 1.0000 | 0.9744 | 0.9750 | 39.3 | +0.0000 |
| 123 | 1.0000 | 0.9950 | 0.9950 | 39.6 | +0.0000 |
| **Mean** | **1.0000** | **0.9898** | | | |
| CI95 | [1.0000, 1.0000] | [0.9560, 1.0235] | | | |

**AUROC:** 1.0000 ± 0.0000 (95% CI: [1.0000, 1.0000])  
**F1:**    0.9898 (95% CI: [0.9560, 1.0235])  
**Δ AUROC vs Q56 CNN baseline:** +0.0000  

## Q56 CNN Baseline Comparison

| Dataset | Q56 CNN AUROC | Q57 DV-QNN AUROC | Δ AUROC | Q57 Params (trainable) |
|---------|--------------|------------------|---------|------------------------|
| VinDr-SpineXR | 0.9731 | 0.8842 | -0.0888 | 574 |
| BUU-LSPINE | 1.0000 | 1.0000 | +0.0000 | 574 |

## Validation Checklist

- [x] Custom DV-QNN head integrated with MobileNetV3-Large feature extractor
- [x] Trained on both datasets (VinDr-SpineXR + BUU-LSPINE) across all seeds
- [x] Leaderboard CSV written with AUROC/F1/CI95/circuit_depth/params/runtime
- [x] Validation report documents DV-QNN results vs CNN baseline
- [x] Only custom in-house quantum framework used (qcore — no PennyLane/Qiskit)

## Notes

- VinDr-SpineXR uses Q49 pre-computed 128-dim embeddings (MobileNetV3-Large + frozen random projection 960→128). QNN head trained fresh per seed.
- BUU-LSPINE uses synthetic procedural spine ROI images (same generator as Q56). Features extracted via MobileNetV3-Large + frozen seeded random projection.
- Trainable parameters per run: DV-QNN head only (574 params). Backbone and projection are fully frozen.
- DV-QNN performs per-sample circuit simulation (CPU-only). Training is slower than classical head but gradient flows end-to-end.
- Reference baselines: Q46 winner AUROC = 0.9873 | Q56 CNN VinDr AUROC = 0.9731 | Q56 CNN BUU AUROC = 1.0000
