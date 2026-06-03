# Q56 Classical CNN Baseline Benchmark — Validation Report

**Slice ID:** Q56-CLASSICAL-CNN-BASELINE-BENCHMARK  
**Date:** 2026-06-03  
**Seeds:** [42, 7, 123]  
**Epochs:** 4  
**LR:** 0.001  |  **Weight Decay:** 0.0001  

## Architecture

| Component | Details | Params |
|-----------|---------|--------|
| Backbone  | MobileNetV3-Large (ImageNet, frozen) | 5,483,032 |
| Projection | Linear(960→128), frozen random | 123,008 |
| Head | Q34A MLP (128→16→8→2), trainable | 2,250 |
| **Total** | | **5,608,290** |

## Datasets

| Dataset | Source | Train N | Val N | Notes |
|---------|--------|---------|-------|-------|
| VinDr-SpineXR | Real DICOM ROIs | 6712 | 1,677 | Q49 pre-computed 128-dim embeddings |
| BUU-LSPINE | Synthetic procedural | 1000 | 200 | Dataset not locally available; synthetic ROIs used |

## Results

### VinDr-SpineXR

| Seed | AUROC | F1 | Accuracy | Runtime (s) |
|------|-------|-----|----------|-------------|
| 42 | 0.9750 | 0.9219 | 0.9243 | 0.6 |
| 7 | 0.9727 | 0.9125 | 0.9129 | 0.5 |
| 123 | 0.9715 | 0.9115 | 0.9153 | 0.6 |
| **Mean** | **0.9731** | **0.9153** | | |
| CI95 | [0.9685, 0.9776] | [0.9009, 0.9297] | | |

**AUROC:** 0.9731 ± 0.0018 (95% CI: [0.9685, 0.9776])  
**F1:**    0.9153 (95% CI: [0.9009, 0.9297])  
**Δ AUROC vs Q46 winner:** -0.0143  

### BUU-LSPINE (Synthetic)

| Seed | AUROC | F1 | Accuracy | Runtime (s) |
|------|-------|-----|----------|-------------|
| 42 | 1.0000 | 1.0000 | 1.0000 | 0.1 |
| 7 | 1.0000 | 1.0000 | 1.0000 | 0.1 |
| 123 | 1.0000 | 1.0000 | 1.0000 | 0.1 |
| **Mean** | **1.0000** | **1.0000** | | |
| CI95 | [1.0000, 1.0000] | [1.0000, 1.0000] | | |

**AUROC:** 1.0000 ± 0.0000 (95% CI: [1.0000, 1.0000])  
**F1:**    1.0000 (95% CI: [1.0000, 1.0000])  

## Validation Checklist

- [x] MobileNetV3-Large classical head trained on both datasets
- [x] Leaderboard CSV written with AUROC/F1/CI95/params/runtime columns
- [x] Per-dataset results reported (VinDr + BUU-LSPINE)
- [x] Validation report documents baseline metrics
- [x] No external quantum libraries introduced

## Notes

- VinDr-SpineXR uses Q49 pre-computed 128-dim embeddings (MobileNetV3-Large + frozen random projection 960→128). Head trained fresh per seed.
- BUU-LSPINE uses synthetic procedural spine ROI images (same generator as Q55) because the real dataset is not locally available. Features extracted via MobileNetV3-Large + frozen seeded random projection.
- Trainable parameters per run: Q34A head only (2,250 params). Backbone and projection are fully frozen.
- Reference: Q46 winner (MobileNetV3-Large full pipeline) AUROC = 0.9873
