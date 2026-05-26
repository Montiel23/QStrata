# VinDr-SpineXR Binary Dataset Loader Report
## Slice Q15

**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-25  
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

Slice Q14 produced the full 10,466-image exported VinDr-SpineXR binary ROI dataset
(`data/processed/vindr_binary_roi_224/`, gitignored) with all 28 validation checks passing.
This slice (Q15) implements a PyTorch `Dataset` class that reads from that manifest and
validates the data loading pipeline end-to-end without performing any model training.

**No model training occurs in this slice.**  
**No QNN work occurs in this slice.**  
**No exported dataset files are modified or committed.**

---

## 2. Dataset Root and Manifest Path

| Parameter | Value |
|---|---|
| Dataset root | `data/processed/vindr_binary_roi_224/` |
| Manifest | `data/processed/vindr_binary_roi_224/manifest.csv` |
| Total manifest rows | 10,466 |
| Manifest columns | 24 |
| Image format | 224×224 grayscale PNG, mode L, uint8 [0, 255] |
| Dataset status | Gitignored — not committed |

---

## 3. Dataset Class Implemented

**File:** `qcore/data/vindr_spinexr.py`  
**Class:** `VinDrSpineXRBinaryDataset`

Follows the existing `qcore/data/` project style:
- `pandas` for manifest CSV reading (consistent with `spine_dataset.py`)
- `PIL.Image` for PNG loading (no OpenCV dependency)
- `torch.tensor()` for float32 tensor construction (avoids `torch.from_numpy()` NumPy C-API version conflict present in the GPU container)
- Namespace packages — no `__init__.py` created or modified

---

## 4. Constructor API

```python
VinDrSpineXRBinaryDataset(
    root="data/processed/vindr_binary_roi_224",
    split="train",
    transform=None,
    target_transform=None,
    return_metadata=False,
)
```

| Parameter | Type | Description |
|---|---|---|
| `root` | `str` | Path to exported dataset root. Must contain `manifest.csv`. |
| `split` | `str` | One of `"train"`, `"val"`, `"test"`. |
| `transform` | `callable` or `None` | Optional transform applied to image tensor after normalisation. |
| `target_transform` | `callable` or `None` | Optional transform applied to label tensor. |
| `return_metadata` | `bool` | If `True`, `__getitem__` returns `(image, label, metadata_dict)`. |

**`__getitem__` return:**
- `return_metadata=False` (default): `(image_tensor, label)` — `(torch.Tensor[1,224,224], torch.long)`
- `return_metadata=True`: `(image_tensor, label, metadata_dict)`

**Additional method:** `class_counts() → dict[int, int]` — returns `{label: count}` for the instantiated split.

---

## 5. Required Manifest Columns Validated

The constructor validates all required columns before any data loading occurs.

| Column | Present in manifest | Status |
|---|---|---|
| `sample_id` | ✅ | PASS |
| `image_id` | ✅ | PASS |
| `split` | ✅ | PASS |
| `binary_label` | ✅ | PASS |
| `original_label` | ✅ | PASS |
| `output_path` | ✅ | PASS |
| `crop_strategy` | ✅ | PASS |
| `has_bbox` | ✅ | PASS |
| `is_pseudo_roi` | ✅ | PASS |
| `background_fraction` | ✅ | PASS |
| `fallback_used` | ✅ | PASS |

**All 11 required columns: PASS**

---

## 6. Constructor Validations Implemented

| Check | Behaviour on failure |
|---|---|
| `root` directory exists | Raises `FileNotFoundError` with descriptive message |
| `manifest.csv` inside `root` exists | Raises `FileNotFoundError` |
| `split` ∈ `{train, val, test}` | Raises `ValueError` |
| All required manifest columns present | Raises `ValueError` listing missing columns |
| All `output_path` files for split exist | Raises `FileNotFoundError` with first 5 missing paths |
| `binary_label` values ⊆ `{0, 1}` | Raises `ValueError` listing unexpected values |

---

## 7. Split Lengths

| Split | Row count | Expected | Status |
|---|---|---|---|
| train | 6,712 | 6,712 | ✅ PASS |
| val | 1,677 | 1,677 | ✅ PASS |
| test | 2,077 | 2,077 | ✅ PASS |

---

## 8. Class Counts by Split

| Split | Label 0 (No Finding) | Label 1 (Any Pathology) | Status |
|---|---|---|---|
| train | 3,408 | 3,304 | ✅ PASS |
| val | 852 | 825 | ✅ PASS |
| test | 1,070 | 1,007 | ✅ PASS |

---

## 9. Sample Tensor Validation

One sample loaded from each split and validated:

| Split | Shape | Dtype | Range | Label value | Status |
|---|---|---|---|---|---|
| train | `[1, 224, 224]` | `torch.float32` | [0.2078, 0.5922] | 1 | ✅ PASS |
| val | `[1, 224, 224]` | `torch.float32` | [0.5020, 1.0000] | 1 | ✅ PASS |
| test | `[1, 224, 224]` | `torch.float32` | [0.3255, 0.9922] | 1 | ✅ PASS |

All tensor ranges are within [0.0, 1.0]. All labels in {0, 1}.

---

## 10. DataLoader Batch Validation

One batch pulled from a `DataLoader` for each split (`batch_size=8`):

| Split | shuffle | Batch image shape | Labels shape | Label values | Status |
|---|---|---|---|---|---|
| train | ✅ True | `(8, 1, 224, 224)` | `(8,)` | ⊆ {0, 1} | ✅ PASS |
| val | ❌ False | `(8, 1, 224, 224)` | `(8,)` | ⊆ {0, 1} | ✅ PASS |
| test | ❌ False | `(8, 1, 224, 224)` | `(8,)` | ⊆ {0, 1} | ✅ PASS |

---

## 11. Metadata Validation

`return_metadata=True` tested on `split="train"`, sample index 0.

Return type: 3-tuple `(image_tensor, label, metadata_dict)` — **PASS**

All required keys present in metadata dict:

| Key | Present | Status |
|---|---|---|
| `sample_id` | ✅ | PASS |
| `image_id` | ✅ | PASS |
| `split` | ✅ | PASS |
| `binary_label` | ✅ | PASS |
| `original_label` | ✅ | PASS |
| `output_path` | ✅ | PASS |
| `crop_strategy` | ✅ | PASS |
| `has_bbox` | ✅ | PASS |
| `is_pseudo_roi` | ✅ | PASS |
| `background_fraction` | ✅ | PASS |
| `fallback_used` | ✅ | PASS |

**Metadata check: PASS**

---

## 12. Guardrails Confirmed

| Guardrail | Status |
|---|---|
| No model training | ✅ Confirmed — smoke test script explicitly excluded model imports |
| No QNN work | ✅ Confirmed — no `qcore/circuit`, `qcore/ansatz`, or quantum code touched |
| No exported dataset modification | ✅ Confirmed — loader is read-only; no files written to `data/processed/` |
| No `data/processed/` files staged | ✅ Confirmed — gitignored by `.gitignore:27:data/processed/` |
| No branch switch | ✅ Confirmed |
| No push | ✅ Confirmed |

---

## 13. Known Limitations

1. **No on-the-fly augmentation built in.** The `transform` parameter accepts any callable, but no augmentation policy is pre-defined in the loader. Augmentation (random flip, rotation, brightness jitter) must be composed externally and passed via `transform` when needed for training.

2. **Single resolution only.** The loader reads from the 224×224 export. A 128×128 export (secondary experimental resolution from Q12) is not yet produced. If needed, a separate export and instantiation with a different `root` path is required.

3. **No multi-crop support.** Each manifest row is one crop. The loader does not support multiple crops per image or test-time augmentation aggregation.

4. **No in-memory caching.** Each `__getitem__` call reads from disk. For long training runs, a caching wrapper or memory-mapped format (e.g. HDF5) may be warranted if I/O becomes a bottleneck.

5. **NumPy 1.x / 2.x C-API warning in container.** The GPU container has NumPy 2.2.6 installed against a PyTorch build targeting NumPy 1.x. This produces a non-fatal `UserWarning` on `torch` import and breaks `torch.from_numpy()`. The loader uses `torch.tensor()` as a copy-based workaround. Resolving the container dependency conflict is deferred to infrastructure work.

---

## 14. Next Slice Recommendation

```
Slice Q16 — VinDr-SpineXR Classical Baseline Smoke Test

Goal:
Run a minimal classical CNN training smoke test using the VinDrSpineXRBinaryDataset
loader to validate end-to-end training mechanics before a full baseline.
```

---

## 15. Smoke Test Output

```
=== VinDr-SpineXR Binary Dataset Loader Smoke Test ===
Root: data/processed/vindr_binary_roi_224

Dataset lengths:
  train: 6712
  val:   1677
  test:  2077

Class counts:
  train: label0=3408, label1=3304
  val:   label0=852,  label1=825
  test:  label0=1070, label1=1007

Sample checks:
  train: image_shape=(1, 224, 224), dtype=torch.float32, range=[0.2078, 0.5922], label=1
  val:   image_shape=(1, 224, 224), dtype=torch.float32, range=[0.5020, 1.0000], label=1
  test:  image_shape=(1, 224, 224), dtype=torch.float32, range=[0.3255, 0.9922], label=1

Batch checks:
  train: batch_shape=(8, 1, 224, 224), labels_shape=(8,)
  val:   batch_shape=(8, 1, 224, 224), labels_shape=(8,)
  test:  batch_shape=(8, 1, 224, 224), labels_shape=(8,)

Metadata check: PASS

Smoke test: PASS
```

---

```
Loader status: PASS
```
