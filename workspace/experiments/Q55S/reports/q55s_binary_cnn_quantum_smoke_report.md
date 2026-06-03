# Q55S — Binary CNN + Custom Quantum Smoke Report
**Date:** 2026-06-03 06:14 UTC
**Runtime:** 2.0s  |  **Limit:** 1800s

---

## Asset Discovery

- Q49 embeddings dir: `/home/mike/research-projects/QStrata/workspace/experiments/Q49/embeddings` — EXISTS
- DV-QNN import (qcore.ansatz.medical_ansatz): OK
- CV-QNN import (qcore.backends.cvBackend): OK

## Data Source

- Source: **Q49_EMBEDDINGS**
- Synthetic fallback used: `False`
- Train: 64 samples  |  Val: 32  |  Test: 32
- Classes seen in train: 2

## Path Status

| Path | Status | Import | Train Supported |
|------|--------|--------|-----------------|
| Classical CNN head | PASS | N/A | Yes |
| Custom DV-QNN | PASS | OK | True |
| Custom CV-QNN | PASS | OK | True |

## Metrics (Test Set)

| Model | Accuracy | F1 | AUROC Avail | AUROC | Train Loss | Params | Runtime |
|-------|----------|----|-------------|-------|------------|--------|---------|
| classical_cnn_head | 0.5000 | 0.0000 | True | 0.6172 | 0.779411 | 4194 | 1.08s |
| custom_dv_qnn | 0.5000 | 0.0000 | True | 0.6289 | 0.733827 | 574 | 0.8s |
| custom_cv_qnn | 0.5938 | 0.4348 | True | 0.5469 | 0.867317 | 532 | 0.16s |

## Warnings

- None

## Hard Failures

- None

## Generated Artifacts

- `workspace/experiments/Q55S/results/q55s_smoke_metrics.csv`
- `workspace/experiments/Q55S/results/q55s_smoke_results.json`
- `workspace/experiments/Q55S/figures/q55s_smoke_metrics.svg`

---

## Recommendation

**READY_FOR_FULL_BINARY_CNN_QUANTUM_CAMPAIGN**

### Next Step
All three paths operational. Proceed with `sliceforge loop --project qstrata --night-auto --max-iterations 10 --budget-aware --allow-benchmarks --notify --yes`
