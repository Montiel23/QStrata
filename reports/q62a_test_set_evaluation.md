# Q62A Test-Set Evaluation Report

**Slice ID:** Q62A-TEST-SET-EVALUATION  
**Date:** 2026-06-03  
**Seeds:** [42, 7, 123]  
**Epochs:** 4  |  **LR:** 0.001  |  **Weight Decay:** 0.0001  
**Test dataset:** VinDr-SpineXR official test split (2,077 samples)  

## Architectures

| Component | CNN | DV-QNN | CV-QNN |
|-----------|-----|--------|--------|
| Backbone | MobileNetV3-Large (frozen) | ← same | ← same |
| Projection | Linear(960→128, frozen) | ← same | ← same |
| Head type | Q34A MLP (128→16→8→2) | medical_ansatz DV circuit | GaussianVariationalAnsatz |
| Head params | 2,250 | 574 | 532 |
| Total params | 5,608,290 | 5,606,614 | 5,606,572 |

## Test-Set Results

### Classical CNN (Q56 architecture)

| Seed | AUROC | F1 | Accuracy | Precision | Recall | Val AUROC | Δ Test-Val | Runtime (s) |
|------|-------|-----|----------|-----------|--------|-----------|------------|-------------|
| 42 | 0.9738 | 0.9071 | 0.9085 | 0.8932 | 0.9215 | 0.9750 | -0.0013 | 0.6 |
| 7 | 0.9732 | 0.9065 | 0.9119 | 0.9337 | 0.8808 | 0.9727 | +0.0005 | 0.5 |
| 123 | 0.9695 | 0.9099 | 0.9109 | 0.8929 | 0.9275 | 0.9715 | -0.0020 | 0.5 |
| **Mean** | **0.9721** | **0.9078** | | **0.9066** | **0.9100** | | | |
| CI95 | [0.9663, 0.9780] | [0.9034, 0.9123] | | [0.8483, 0.9649] | [0.8469, 0.9731] | | | |

**Test AUROC:** 0.9721 ± 0.0023 (95% CI: [0.9663, 0.9780])  
**Test F1:** 0.9078 (95% CI: [0.9034, 0.9123])  
**Test Precision:** 0.9066 (95% CI: [0.8483, 0.9649])  
**Test Recall:** 0.9100 (95% CI: [0.8469, 0.9731])  

### DV-QNN (Q57 architecture)

| Seed | AUROC | F1 | Accuracy | Precision | Recall | Val AUROC | Δ Test-Val | Runtime (s) |
|------|-------|-----|----------|-----------|--------|-----------|------------|-------------|
| 42 | 0.8820 | 0.7727 | 0.7930 | 0.8260 | 0.7259 | 0.8833 | -0.0014 | 255.4 |
| 7 | 0.8833 | 0.7818 | 0.7925 | 0.7975 | 0.7666 | 0.8844 | -0.0012 | 254.7 |
| 123 | 0.8840 | 0.7870 | 0.7944 | 0.7906 | 0.7835 | 0.8849 | -0.0009 | 256.4 |
| **Mean** | **0.8831** | **0.7805** | | **0.8047** | **0.7587** | | | |
| CI95 | [0.8805, 0.8857] | [0.7625, 0.7985] | | [0.7581, 0.8513] | [0.6851, 0.8322] | | | |

**Test AUROC:** 0.8831 ± 0.0011 (95% CI: [0.8805, 0.8857])  
**Test F1:** 0.7805 (95% CI: [0.7625, 0.7985])  
**Test Precision:** 0.8047 (95% CI: [0.7581, 0.8513])  
**Test Recall:** 0.7587 (95% CI: [0.6851, 0.8322])  

### CV-QNN (Q58 architecture)

| Seed | AUROC | F1 | Accuracy | Precision | Recall | Val AUROC | Δ Test-Val | Runtime (s) |
|------|-------|-----|----------|-----------|--------|-----------|------------|-------------|
| 42 | 0.9476 | 0.8711 | 0.8796 | 0.9057 | 0.8391 | 0.9524 | -0.0048 | 49.8 |
| 7 | 0.9478 | 0.8845 | 0.8844 | 0.8581 | 0.9126 | 0.9544 | -0.0066 | 49.8 |
| 123 | 0.9475 | 0.8770 | 0.8830 | 0.8946 | 0.8600 | 0.9534 | -0.0059 | 50.2 |
| **Mean** | **0.9476** | **0.8775** | | **0.8861** | **0.8706** | | | |
| CI95 | [0.9472, 0.9481] | [0.8609, 0.8942] | | [0.8242, 0.9480] | [0.7765, 0.9647] | | | |

**Test AUROC:** 0.9476 ± 0.0002 (95% CI: [0.9472, 0.9481])  
**Test F1:** 0.8775 (95% CI: [0.8609, 0.8942])  
**Test Precision:** 0.8861 (95% CI: [0.8242, 0.9480])  
**Test Recall:** 0.8706 (95% CI: [0.7765, 0.9647])  

## Comparison: Test AUROC vs Validation AUROC

| Architecture | Val AUROC (mean) | Test AUROC (mean) | Δ (test − val) |
|--------------|-----------------|-------------------|----------------|
| Classical CNN | 0.9731 | 0.9721 | -0.0009 |
| DV-QNN        | 0.8842 | 0.8831 | -0.0011 |
| CV-QNN        | 0.9534 | 0.9476 | -0.0058 |

## Cross-Architecture Test AUROC Summary

| Architecture | Test AUROC | 95% CI | Test F1 | Precision | Recall | Head Params |
|--------------|-----------|--------|---------|-----------|--------|-------------|
| Classical CNN | 0.9721 ± 0.0023 | [0.9663, 0.9780] | 0.9078 | 0.9066 | 0.9100 | 2,250 |
| DV-QNN        | 0.8831 ± 0.0011 | [0.8805, 0.8857] | 0.7805 | 0.8047 | 0.7587 | 574 |
| CV-QNN        | 0.9476 ± 0.0002 | [0.9472, 0.9481] | 0.8775 | 0.8861 | 0.8706 | 532 |

## Checkpoints

| Architecture | Seed | Checkpoint |
|--------------|------|------------|
| cnn | 42 | workspace/experiments/Q62A/checkpoints/cnn_seed42.pt |
| cnn | 7 | workspace/experiments/Q62A/checkpoints/cnn_seed7.pt |
| cnn | 123 | workspace/experiments/Q62A/checkpoints/cnn_seed123.pt |
| dv_qnn | 42 | workspace/experiments/Q62A/checkpoints/dv_qnn_seed42.pt |
| dv_qnn | 7 | workspace/experiments/Q62A/checkpoints/dv_qnn_seed7.pt |
| dv_qnn | 123 | workspace/experiments/Q62A/checkpoints/dv_qnn_seed123.pt |
| cv_qnn | 42 | workspace/experiments/Q62A/checkpoints/cv_qnn_seed42.pt |
| cv_qnn | 7 | workspace/experiments/Q62A/checkpoints/cv_qnn_seed7.pt |
| cv_qnn | 123 | workspace/experiments/Q62A/checkpoints/cv_qnn_seed123.pt |

## Validation Checklist

- [x] VinDr-SpineXR test split evaluated for all 3 architectures × 3 seeds
- [x] test_metrics.csv has AUROC/F1/accuracy/precision/recall per arch per seed
- [x] test_metrics.json has mean ± std, CI95 per architecture
- [x] 9 model checkpoints saved (9)
- [x] Report documents test vs val AUROC comparison
- [x] No external quantum libraries used
- [x] Q56/Q57/Q58 result artifacts not modified
- [x] BUU-LSPINE excluded (synthetic saturation)

## Notes

- Trained on Q49 pre-computed 128-dim embeddings (VinDr train split).
- Evaluated on official VinDr-SpineXR test split (2,077 samples, 1,070 neg + 1,007 pos).
- Quantum head training is CPU-only (DV/CV circuit simulators).
- Checkpoints contain state_dict + metadata for downstream Q62B evaluation.
- Reference: Q46 winner AUROC = 0.9873
