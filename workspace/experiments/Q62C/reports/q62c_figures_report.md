# Q62C Figures Report

**Slice ID:** Q62C-ROC-PR-CONFUSION-MATRIX-GENERATION  
**Date:** 2026-06-03  
**Source:** Q62B probability exports - VinDr-SpineXR test split  

## AUROC: Q62C vs Q62A

| Architecture | Q62C AUROC | Q62A AUROC | Delta |
|--------------|-----------|-----------|-------|
| CNN | 0.9721 +/- 0.0019 | 0.9721 +/- 0.0023 | +0.0 |
| DV-QNN | 0.8831 +/- 0.0009 | 0.8831 +/- 0.0011 | -0.0 |
| CV-QNN | 0.9476 +/- 0.0001 | 0.9476 +/- 0.0002 | +0.0 |

## Confusion Matrix Summary

| Architecture | TN | FP | FN | TP | Accuracy | F1 | Precision | Recall |
|--------------|----|----|----|----|----------|----|-----------|--------|
| CNN | 975 | 95 | 91 | 916 | 0.9104 | 0.9079 | 0.9058 | 0.91 |
| DV-QNN | 884 | 186 | 243 | 764 | 0.7933 | 0.7807 | 0.8039 | 0.7587 |
| CV-QNN | 956 | 114 | 130 | 877 | 0.8824 | 0.8777 | 0.8849 | 0.8706 |

## Pass Criteria

- [x] All 6 SVG figures generated
- [x] AUROC matches Q62A
- [x] No training executed
