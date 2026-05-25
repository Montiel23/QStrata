# NAS Candidate Config Mapping v0

- **Status:** Active
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Informed by:** docs/design/nas_candidate_grid_v0.md, Slice 17

---

## Purpose

This document maps each NAS v0 candidate ID to its corresponding YAML config file. It serves as the authoritative reference for any future benchmark sweep or NAS implementation that needs to locate a candidate's config. All six candidates in the approved grid are covered.

---

## Candidate Config Mapping

| Candidate ID | block_type | conv_channels | Config path |
|---|---|---|---|
| C001 | standard | [32, 64] | `experiments/configs/binary_baseline.yaml` |
| C002 | standard | [48, 96] | `experiments/configs/nas_v0_c002_standard_48_96.yaml` |
| C003 | standard | [64, 128] | `experiments/configs/nas_v0_c003_standard_64_128.yaml` |
| C004 | depthwise_sep | [32, 64] | `experiments/configs/binary_baseline_depthwise_sep.yaml` |
| C005 | depthwise_sep | [48, 96] | `experiments/configs/nas_v0_c005_depthwise_sep_48_96.yaml` |
| C006 | depthwise_sep | [64, 128] | `experiments/configs/binary_baseline_depthwise_sep_wide.yaml` |

---

## Notes

- C001, C004, and C006 use existing validated configs and have prior benchmark results from Slices 12 and 13
- C002, C003, and C005 are new configs created in Slice 17 and have not yet been benchmarked
- All configs share fixed training settings — dropout, batch_norm, optimizer, learning rate, epochs, class weights, dataset, and input shape — as defined in `docs/design/nas_search_space_v0.md`
- This mapping is the input reference for the next human-approved benchmark or NAS slice
