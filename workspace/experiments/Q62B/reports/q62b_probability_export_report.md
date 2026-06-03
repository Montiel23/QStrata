# Q62B Probability Export Report

**Slice ID:** Q62B-PROBABILITY-EXPORT  
**Date:** 2026-06-03  
**Seeds:** [42, 7, 123]  
**Checkpoints from:** Q62A (no re-training)  
**Total runtime:** 73.6s (1.2 min)  

## Exported Files

| Architecture | Split | Rows | Samples × Seeds | CSV |
|-------------|-------|------|-----------------|-----|
| cnn | test | 6231 | 2077 × 3 | `cnn_vindr_test_probs.csv` |
| cnn | val  | 5031 | 1677 × 3 | `cnn_vindr_val_probs.csv` |
| dv_qnn | test | 6231 | 2077 × 3 | `dv_qnn_vindr_test_probs.csv` |
| dv_qnn | val  | 5031 | 1677 × 3 | `dv_qnn_vindr_val_probs.csv` |
| cv_qnn | test | 6231 | 2077 × 3 | `cv_qnn_vindr_test_probs.csv` |
| cv_qnn | val  | 5031 | 1677 × 3 | `cv_qnn_vindr_val_probs.csv` |

## CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| sample_id | int | 0-based index into the split |
| y_true | int | Ground-truth label (0=normal, 1=abnormal) |
| y_score | float | Softmax probability of positive class ∈ [0,1] |
| y_pred | int | Argmax prediction (0 or 1) |
| seed | int | Training seed (42, 7, or 123) |

## Validation Checklist

- [x] CNN probabilities exported (test + val)
- [x] DV-QNN probabilities exported (test + val)
- [x] CV-QNN probabilities exported (test + val)
- [x] y_score values are softmax probabilities ∈ [0,1]
- [x] CSV schema: sample_id, y_true, y_score, y_pred, seed
- [x] Q62A checkpoints used (no re-training)
- [x] No external quantum libraries used

## Notes

- Inference only — model weights are frozen checkpoints from Q62A.
- Each CSV contains rows for all seeds (seed column distinguishes them).
- Q62C will use these files to generate ROC/PR curves and confusion matrices.
