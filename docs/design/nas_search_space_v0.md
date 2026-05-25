# NAS Search Space v0 — Design Document

- **Status:** Draft
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Informed by:** ADR-002, Slices 11–14

---

## Purpose

This document defines a constrained, evidence-based CNN search space for future NSGA-II exploration on PneumoniaMNIST. The search space is deliberately small because the current problem scale is small — binary classification on 28×28 grayscale images — and an oversized space would be unjustified by the available evidence. Every candidate decision recorded here is grounded in manual benchmark results from Slices 11–14; no block type or channel configuration is included speculatively. No NAS implementation is part of this slice; this document is design and documentation only.

---

## Candidate Block Types

**Accepted for search space:**
- `standard` — proven baseline; serves as reference and valid search candidate
- `depthwise_sep` — recovered accuracy within tolerance at `[64,128]`; channel width is a required searchable dimension

**Explicitly excluded from v0 search space:**
- `asymmetric` — not yet manually validated
- `grouped_shuffle` — not yet manually validated
- `dilated` — not yet manually validated

Excluded block types may be considered for inclusion in a future search space version after manual benchmark validation equivalent to the process followed in Slices 12–13.

---

## Search Variables v0

**block_type**
- Candidates: `standard`, `depthwise_sep`
- Status: searchable in v0

**conv_channels**
- Candidates: `[32, 64]`, `[48, 96]`, `[64, 128]`
- Status: searchable in v0
- Note: channel width is the primary capacity dial for `depthwise_sep`; its effect on accuracy was the subject of Slice 13

**dropout**
- Status: fixed at current baseline value for v0
- Rationale: isolate block type and channel width as the primary variables; introduce dropout as a search dimension in a later version if warranted

**batch_norm**
- Status: fixed at current baseline behavior for v0
- Rationale: same as dropout; avoid compounding variables in the first search

**optimizer, learning rate, epochs, class weights**
- Status: all fixed at current baseline values for v0
- Rationale: training settings are not under evaluation; fixing them ensures any accuracy differences are attributable to architecture choices

---

## Objectives for Future NAS

The following objectives are defined here for future implementation but are not implemented in this slice:

1. **Maximize validation accuracy** — primary objective; test accuracy is reserved for final evaluation only and must not be used during search
2. **Minimize parameter count** — efficiency objective; prefer smaller models when accuracy is comparable
3. **Minimize mean inference latency** — runtime efficiency objective; measured as ms/batch on the GPU stack

These three objectives make this a multi-objective search problem suitable for NSGA-II.

---

## Constraints

- No candidate architecture in the search space should exceed the parameter count of `standard [32,64]` (19,138 params) by more than 2×
- Training budget per candidate must remain small; epoch count is fixed at the baseline value
- Validation accuracy is the fitness signal during search; test accuracy is held out
- The search space must remain enumerable manually before any automated search is run; if the full grid cannot be listed in a table, the space is too large for v0

---

## Out of Scope for This Document

- pymoo installation or configuration
- NSGA-II implementation
- Any new experiments or training runs
- Any new block type implementations
- Any new configs beyond what already exists
- Notebook modifications
- Docker modifications
- QML integration

---

## Next Step Recommendation

The next slice should produce a small, enumerable candidate grid — a flat list of every architecture combination implied by the v0 search variables defined above — so that the complete search space can be reviewed by the human architect before any automated search begins. That next slice is subject to human approval before it proceeds. No implementation work of any kind should begin until the candidate grid has been reviewed and approved.
