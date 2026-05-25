# NAS Candidate Grid v0

- **Status:** Draft
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Informed by:** docs/design/nas_search_space_v0.md, ADR-002, Slices 11–15

---

## Purpose

This document enumerates the complete candidate grid implied by the v0 search space so that the human architect can review the full set of candidates before any automation begins. The grid is the Cartesian product of `block_type` and `conv_channels` as defined in `nas_search_space_v0.md`. It is intentionally small — 6 candidates — appropriate for the current dataset and model scale (PneumoniaMNIST, binary, 28×28 grayscale). No NAS implementation is part of this slice; this document is design and documentation only.

---

## Candidate Grid

| Candidate ID | block_type | conv_channels |
|---|---|---|
| C001 | standard | [32, 64] |
| C002 | standard | [48, 96] |
| C003 | standard | [64, 128] |
| C004 | depthwise_sep | [32, 64] |
| C005 | depthwise_sep | [48, 96] |
| C006 | depthwise_sep | [64, 128] |

C001 is the existing validated baseline (`binary_baseline.yaml`). C004 and C006 have prior benchmark results from Slices 12 and 13. C002, C003, and C005 are untested candidates.

---

## Fixed Training Settings

All of the following remain fixed at their current baseline values for every candidate in this grid:

- `dropout` — fixed at baseline value
- `batch_norm` — fixed at baseline behavior
- `optimizer` — fixed at baseline value
- `learning rate` — fixed at baseline value
- `epochs` — fixed at baseline value
- `class weights` — fixed at baseline value
- `dataset` — PneumoniaMNIST, binary classification, 28×28 grayscale
- `input shape` — `(batch, 1, 28, 28)`

Fixing these variables ensures that any differences in measured outcomes between candidates are attributable solely to `block_type` and `conv_channels`.

---

## Future NAS Objectives

The following objectives are defined here for future implementation and are not implemented in this slice:

1. **Maximize validation accuracy** — validation accuracy is the fitness signal during search; test accuracy is reserved for final evaluation only
2. **Minimize parameter count** — prefer architecturally smaller models when accuracy is comparable
3. **Minimize mean inference latency** — measured as ms/batch on the GPU stack

These three objectives constitute a multi-objective optimization problem suitable for NSGA-II.

---

## Search Constraints

- Test accuracy must not be used during NAS search; it is reserved for final candidate evaluation only
- Validation accuracy is the sole fitness signal during search
- The candidate grid must remain fully enumerable; if the grid cannot be listed in a table, it is too large for v0
- No candidate should exceed the parameter budget ceiling defined in `nas_search_space_v0.md`

---

## Out of Scope

- NAS implementation of any kind
- pymoo installation or configuration
- NSGA-II code
- Any new experiments or training runs
- Config files for the six candidates
- Any new block type implementations
- Notebook modifications
- Docker modifications
- Any code changes

---

## Next Step Recommendation

The next human-approved slice should create YAML config files for the six candidates in this grid — one config per candidate, following the naming and structural conventions of the existing baseline configs. After those configs exist and have been approved, a later human-approved slice may benchmark them. No benchmarking or NAS implementation should begin until the config files have been reviewed and approved by the human architect.
