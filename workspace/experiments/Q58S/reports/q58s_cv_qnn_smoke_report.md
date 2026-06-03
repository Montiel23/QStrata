# Q58S — Custom CV-QNN Smoke Test Report
**Date:** 2026-06-03 16:35 UTC
**Runtime:** 3.2s  |  **Limit:** 60s  |  **Seed:** 42

---

## Asset Discovery

- Q49 embeddings dir: `/home/mike/research-projects/QStrata/workspace/experiments/Q49/embeddings` — EXISTS
- GaussianBackend import: OK
- cv_spine_ansatz import: OK

## Data Source

- Source: **Q49_EMBEDDINGS**
- Synthetic fallback used: `False`
- Train: 64  |  Val: 32  |  Test: 32
- Classes in train: 2

## Circuit Manifest

| Parameter | Value |
|-----------|-------|
| n_modes | 2 |
| depth | 1 |
| squeezing_cap | 1.5 |
| Gate sequence | Displacement → Squeezing → Rotation → Beamsplitter (ring) |
| Data reuploading | Per depth layer (encoded_input added to mu) |
| Measurement | Homodyne X-quadrature |
| Squeezing bound | tanh(r_raw) × squeezing_cap — PASS |
| Total trainable params | 532 |

## CV-QNN Status

| Check | Status |
|-------|--------|
| Import | PASS |
| Forward pass | PASS |
| Training (1 epoch) | PASS |
| Evaluation | PASS |

## Metrics (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | 0.59375 |
| F1 | 0.5517241379310345 |
| AUROC | 0.5898 |
| Train loss | 0.93577 |
| Runtime | 3.14s |

## Warnings

- SVG generation failed: No module named 'matplotlib'

## Hard Failures

- None

## Generated Artifacts

- `workspace/experiments/Q58S/results/q58s_smoke_metrics.csv`
- `workspace/experiments/Q58S/results/q58s_smoke_results.json`

---

## Recommendation

**READY_FOR_Q58_FULL_BENCHMARK**

CV-QNN pipeline fully operational. Proceed with `sliceforge execute --project qstrata --slice-id Q58-CUSTOM-CV-QNN-BINARY-BENCHMARK --night-auto --yes --allow-benchmarks`
