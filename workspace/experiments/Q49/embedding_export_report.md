# Q49 Embedding Export Report

**Slice**: Q49-EMBEDDING-EXPORT-PIPELINE  
**Date**: 2026-06-02 18:00 UTC  
**Backbone**: MobileNetV3-Large (IMAGENET1K_V1, frozen)  
**Projection**: Linear(960 -> 128), frozen, seed=42  
**Embedding dim**: 128  
**Device**: cuda  
**Wall time**: 50.9s  

## Split Summary

| Split | N | Class 0 | Class 1 | pos% | Shape |
|-------|---|---------|---------|------|-------|
| train | 6712 | 3408 | 3304 | 49.2% | (6712, 128) |
| val | 1677 | 852 | 825 | 49.2% | (1677, 128) |
| test | 2077 | 1070 | 1007 | 48.5% | (2077, 128) |

## Dimensionality Verification

- **train**: shape=(6712, 128) -> PASS
- **val**: shape=(1677, 128) -> PASS
- **test**: shape=(2077, 128) -> PASS

**Check**: ALL PASS

## Output Files

- `workspace/experiments/Q49/embeddings/train_embeddings.npy` (3.277 MB)
- `workspace/experiments/Q49/embeddings/train_labels.npy` (0.051 MB)
- `workspace/experiments/Q49/embeddings/val_embeddings.npy` (0.819 MB)
- `workspace/experiments/Q49/embeddings/val_labels.npy` (0.013 MB)
- `workspace/experiments/Q49/embeddings/test_embeddings.npy` (1.014 MB)
- `workspace/experiments/Q49/embeddings/test_labels.npy` (0.016 MB)

## Validation

- shape (N,128): PASS
- Backbone frozen: True
- Preprocessing: CLAHE(clip=3.0, tile=(4, 4)) + 3ch + ImageNet norm
- shuffle=False
