# Q66P — Pure Quantum Readout Smoke and Full Campaign Plan

**Slice ID:** Q66P-PURE-QUANTUM-READOUT-SMOKE-AND-FULL-CAMPAIGN-PLAN  
**Date:** 2026-06-04  
**Mode:** Documentation  
**Status:** COMPLETE  

---

## Summary

Created two execution campaigns for the pure quantum readout study:

1. **`pure_quantum_readout_smoke`** — ACTIVE: validates end-to-end pure readout execution on 64/32/32 samples, 1 epoch, 1 seed
2. **`pure_quantum_readout_full`** — PLANNED: full 3-seed benchmarks, metrics extraction, ablation, comparison, and final report

Gate: Q74S must emit `READY_FOR_FULL_PURE_QUANTUM_READOUT_CAMPAIGN` before the full campaign unlocks.

---

## Smoke Campaign: `pure_quantum_readout_smoke`

| Slice | Name | Status | Depends On |
|-------|------|--------|------------|
| Q66 | Pure Quantum Visualization Package | READY | — |
| Q67S | DV-QNN Pure Readout Smoke | BLOCKED | Q66 |
| Q68S | CV-QNN Single Homodyne Readout Smoke | BLOCKED | Q66 |
| Q68DS | CV-QNN Dual Homodyne Readout Smoke | BLOCKED | Q66 |
| Q72S | Quantum Metrics Smoke | BLOCKED | Q67S, Q68S, Q68DS |
| Q74S | Hybrid vs Pure Smoke Comparison | BLOCKED (GATE) | Q72S |

Smoke constraints: max 64 train / 32 val / 32 test samples, 1 seed (42), 1 epoch, VinDr-SpineXR only.

**Validation result:** PASS (5 PASS, 1 WARN on Q74S for benchmark runtime)

---

## Full Campaign: `pure_quantum_readout_full`

| Slice | Name | Status | Depends On |
|-------|------|--------|------------|
| Q69 | DV-QNN Pure Readout Full Benchmark | BLOCKED | Q74S |
| Q70 | CV-QNN Single Homodyne Readout Full Benchmark | BLOCKED | Q74S |
| Q71 | CV-QNN Dual Homodyne Readout Full Benchmark | BLOCKED | Q74S |
| Q72 | Quantum Metrics Extraction | BLOCKED | Q69, Q70, Q71 |
| Q73 | Noise Detector Circuit Ablation | BLOCKED | Q72 |
| Q74 | Hybrid vs Pure Quantum Comparison | BLOCKED | Q73 |
| Q75 | Pure Quantum Readout Final Report | BLOCKED | Q74 |

Full constraints: seeds [42, 7, 123], VinDr test evaluation, save y_true/y_score/y_pred, SVG figures, LaTeX tables.

**Validation result:** PASS (2 PASS, 5 WARN for benchmark runtime — expected)

---

## Dependency Graph

### Smoke Campaign

```
Q66
 ├── Q67S ──┐
 ├── Q68S ──┼── Q72S ── Q74S [GATE]
 └── Q68DS ─┘
```

### Full Campaign

```
Q74S [GATE]
 ├── Q69 ──┐
 ├── Q70 ──┼── Q72 ── Q73 ── Q74 ── Q75
 └── Q71 ──┘
```

---

## Validation Commands Results

```
sliceforge campaign validate --project qstrata --campaign-id pure_quantum_readout_smoke
→ PASS (5 PASS, 1 WARN)

sliceforge campaign validate --project qstrata --campaign-id pure_quantum_readout_full
→ PASS (2 PASS, 5 WARN)

sliceforge queue --project qstrata --allow-benchmarks
→ Queue: [Q55, Q66] — Q66 is first new-campaign slice

sliceforge loop --project qstrata --night-auto --max-iterations 10 --budget-aware --allow-benchmarks --notify --yes --dry-run
→ Dry-run queue: [Q55, Q66] — campaign starts with Q66
```

Note: Q55 appears before Q66 in the unconstrained queue because it belongs to the pre-existing `binary_classical_publication_package` campaign (PLANNED status) and was registered earlier in the backlog. Q66 is the first new-campaign slice and the intended start of the pure quantum readout study.

---

## Artifacts Created

| File | Description |
|------|-------------|
| `workspace/projects/qstrata/campaigns/pure_quantum_readout_smoke.yaml` | Smoke campaign definition |
| `workspace/projects/qstrata/campaigns/pure_quantum_readout_full.yaml` | Full campaign definition |
| `workspace/projects/qstrata/backlog.json` | Updated with 13 new slices (Q66–Q75 + Q67S/Q68S/Q68DS/Q72S/Q74S) |
| `workspace/projects/qstrata/inputs/q66_*.md` | Q66 visualization input spec |
| `workspace/projects/qstrata/inputs/q67s_*.md` | Q67S DV-QNN pure readout smoke spec |
| `workspace/projects/qstrata/inputs/q68s_*.md` | Q68S CV-QNN single homodyne smoke spec |
| `workspace/projects/qstrata/inputs/q68ds_*.md` | Q68DS CV-QNN dual homodyne smoke spec |
| `workspace/projects/qstrata/inputs/q72s_*.md` | Q72S quantum metrics smoke spec |
| `workspace/projects/qstrata/inputs/q74s_*.md` | Q74S gate comparison spec |
| `workspace/projects/qstrata/inputs/q69_*.md` | Q69 DV-QNN full benchmark spec |
| `workspace/projects/qstrata/inputs/q70_*.md` | Q70 CV-QNN single homodyne full spec |
| `workspace/projects/qstrata/inputs/q71_*.md` | Q71 CV-QNN dual homodyne full spec |
| `workspace/projects/qstrata/inputs/q72_*.md` | Q72 quantum metrics extraction spec |
| `workspace/projects/qstrata/inputs/q73_*.md` | Q73 noise/detector ablation spec |
| `workspace/projects/qstrata/inputs/q74_*.md` | Q74 hybrid vs pure comparison spec |
| `workspace/projects/qstrata/inputs/q75_*.md` | Q75 final report spec |
