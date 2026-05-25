# Slice 31 — C006-D040 Manual Search Plan

- **Status:** Active
- **Date:** 2026-05-24
- **Branch:** develop
- **Informed by:** Slice 29 stability validation, Slice 29B latency sanity check, Slice 30 dropout expansion

---

## 1. Current Anchor Summary

C006-D040 is the current practical candidate following the Slice 30 dropout expansion. It replaces C006-D030 as the recommended follow-up candidate after passing all gate criteria.

| Field | Value |
|---|---|
| Candidate ID | C006-D040 |
| block_type | depthwise_sep |
| conv_channels | [64, 128] |
| dropout | 0.40 |
| params | 9,870 |
| best_val_acc | 91.79% |
| test_acc (analysis only) | 86.86% |
| latency (ms/batch) | 0.474 |

C006-D040 achieves 91.79% best validation accuracy at 9,870 parameters — a strong accuracy-efficiency profile that sits within 0.77 pp of the v1 Pareto accuracy anchor (C001 at 92.56%) while using roughly half the parameters. It carries slightly higher dropout regularization (0.40 vs 0.30 baseline), which appears to modestly improve generalization at this model size.

---

## 2. Lessons Learned from Slices 29, 29B, and 30

- **Slice 29 (stability validation):** The multi-seed sweep (seeds 42, 7, 123, 999) returned verdict "Not stable enough; stop and investigate" due to a latency std % of mean of 21.4%, exceeding the 15% gate. Importantly, all three validation accuracy gate criteria passed — std(val_acc) = 0.59%, no seed more than 0.67% below mean. The failure was entirely attributable to the latency measurement, not to training instability.

- **Slice 29B (latency sanity check):** GPU-synchronized block timing (`torch.cuda.synchronize()` before and after each block of 25 forward passes, latency = block time ÷ 25) reduced latency std % to **4.9%**, well within the 15% gate. This confirmed that the Slice 29 latency failure was an artifact of naive wall-clock timing around asynchronous CUDA calls — the GPU's queued execution was captured inconsistently without synchronization. C006 was cleared of the latency concern and the verdict updated accordingly.

- **Slice 30 (dropout expansion):** A controlled single-seed (42) sweep across dropout values 0.20, 0.30, and 0.40 showed that all three variants achieve the same best val acc (91.79%–91.98%) and identical parameter counts (9,870). C006-D020 failed the latency gate due to a first-run GPU warmup artifact (49.6% delta), not genuine inference instability. C006-D040 passed all gate criteria with latency delta of 0.9% versus baseline. Higher dropout (0.40) was marginally preferred — it converged to the same accuracy as D020 while showing more stable latency, making it the cleaner candidate for continuation.

---

## 3. Candidate Expansion Options Considered

| Variable | Rationale for | Rationale against | Recommended |
|---|---|---|---|
| weight_decay | Directly targets generalization without affecting architecture, parameter count, or latency; orthogonal to dropout; well-evidenced in small-model settings | No strong signal yet that overfitting is the limiting factor; single-seed result may confound regularization benefit with seed variance | **Yes** |
| learning_rate | Can meaningfully improve convergence stability and final accuracy; easy to sweep | Interacts with the Adam optimizer's adaptive rates in complex ways; changes training dynamics more broadly and makes results harder to isolate | No |
| conv_width | Increasing channels (e.g. [96, 192]) could close the remaining 0.77 pp accuracy gap to C001 | Changes parameter count and latency, breaking the single-variable constraint; constitutes architecture search, not hyperparameter tuning | No |
| batch_size | Can affect generalization through gradient noise characteristics; easy to vary | Changes effective learning rate implicitly; conflates optimization dynamics with regularization; harder to interpret cleanly | No |

---

## 4. One Recommended Next Variable

**Recommended variable: `weight_decay`**

---

## 5. Justification

Weight decay (L2 regularization applied via the optimizer's `weight_decay` parameter) is the appropriate next variable because it directly targets generalization without altering the model architecture, parameter count, or inference latency. Since dropout (0.40) is already relatively high for a 9,870-parameter model, adding a small amount of L2 regularization alongside it may further reduce overfitting and stabilize the validation accuracy across epochs — particularly relevant given that C006-D040's final train accuracy (91.02%) sits modestly above its best validation accuracy (91.79%), suggesting mild generalization headroom. Weight decay affects only the optimizer during training; the model graph, parameter count, and forward-pass latency are identical across all variants, making it a clean single-variable test fully consistent with the controlled manual search strategy. The default Adam optimizer used by the evaluator sets `weight_decay=0.0`; testing 1e-4 and 5e-4 covers the typical useful range for small medical imaging classifiers.

---

## 6. Proposed Candidate Set for Slice 32

| Candidate ID | block_type | conv_channels | dropout | weight_decay | Role |
|---|---|---|---|---|---|
| C006-D040-WD0000 | depthwise_sep | [64, 128] | 0.40 | 0.0 | Variant |
| C006-D040-WD0001 | depthwise_sep | [64, 128] | 0.40 | 1e-4 | Variant |
| C006-D040-WD0005 | depthwise_sep | [64, 128] | 0.40 | 5e-4 | Variant |

Architecture, dropout, and params must remain fixed across all three. Only `weight_decay` changes. C006-D040 (weight_decay=0.0, the current practical candidate) serves as the implicit baseline reference; C006-D040-WD0000 is its explicit controlled equivalent in the new sweep.

---

## 7. Metrics to Collect in Slice 32

| Metric | Role |
|---|---|
| candidate_id | Identity |
| weight_decay | Variable under test |
| params | Sanity check — must equal 9,870 |
| best_val_acc | Primary fitness signal |
| best_epoch | Training behaviour |
| final_train_acc | Overfitting indicator |
| test_acc | Analysis only — not fitness |
| mean_epoch_time | Runtime tracking |
| latency_ms | Must not degrade >15% vs C006-D040 (0.474 ms/batch) |

---

## 8. Decision Gate for Slice 32

A weight decay variant may replace C006-D040 only if ALL of the following are true:

| Criterion | Threshold |
|---|---|
| best_val_acc | Within 0.5 pp of C006-D040 OR better (≥ 91.29%) |
| params | Equal to C006-D040 (9,870) |
| latency delta | ≤ 15% degradation versus C006-D040 (0.474 ms/batch) |
| training | Completes successfully — no failures |

Test accuracy is analysis-only. It must not be used as a fitness signal or gate criterion.

If no variant satisfies all conditions, keep C006-D040 as the practical candidate.

---

## 9. Guardrails

The following constraints are hard limits for Slice 32 and must not be violated:

- `weight_decay` is the only variable that may change across candidates
- No architecture changes — `block_type: depthwise_sep`, `conv_channels: [64, 128]` must remain fixed
- No new block types or conv widths
- No dropout changes — must remain 0.40 across all candidates
- No multi-variable search — one variable, one sweep
- No multi-seed validation — single seed only (seed 42)
- No NAS, no NSGA-II, no pymoo, no Ray, no cloud
- No dashboards, no MLflow, no monitoring stack
- Do NOT modify `qcore/nas/evaluator.py` or any other source file
- No new config files unless unavoidable — prefer script-level `weight_decay` overrides to the optimizer; stop and report before creating any new YAML configs
- Test accuracy is analysis-only — never used as gate criterion or fitness signal

---

## 10. Explicit Non-Goals

- This plan does not expand to multi-seed robustness validation.
- This plan does not introduce architecture search or NAS.
- This plan does not change the baseline dataset, training protocol, or stable benchmark protocol v1.
- This plan does not include inference optimisation, quantisation, or model pruning.
- This plan does not begin implementation — Slice 31 is planning only; no runner script, no configs, no experiment is created here.

---

## 11. Recommended Slice 32 Execution Objective

> **Slice 32 — C006-D040 Weight Decay Expansion:** Run a controlled single-seed (seed 42) training sweep across C006-D040-WD0000, C006-D040-WD0001, and C006-D040-WD0005. Override only the `weight_decay` parameter in the Adam optimizer; hold all other hyperparameters and the architecture fixed. Collect standard per-candidate metrics using the existing evaluator interface. Apply the decision gate defined in this plan — a variant replaces C006-D040 only if all four criteria pass. Emit the exact binary verdict. Commit the runner script and results report in separate logical commits on `develop`. Do not push, merge, or switch branches.
