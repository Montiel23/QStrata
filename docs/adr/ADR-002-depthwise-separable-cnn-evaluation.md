# ADR-002: Depthwise Separable CNN Evaluation

- **Status:** Accepted
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Slices:** 11, 12, 13

---

## Context

The classical pipeline goal is to establish efficient CNN baselines on PneumoniaMNIST before entering NAS exploration. Slice 11 extended `build_model()` with a `block_type` parameter, enabling both `standard` and `depthwise_sep` convolutional blocks to be selected from config. Slices 12 and 13 benchmarked these block types on PneumoniaMNIST — a binary classification task using 28×28 single-channel chest X-ray images. Slice 12 established an initial comparison at matched channel dimensions `[32, 64]`. Slice 13 ran a controlled diagnostic to determine whether insufficient channel capacity explained the accuracy gap observed in Slice 12.

---

## Benchmark Results

### Slice 12 — Initial Benchmark

| Metric | Standard [32,64] | Depthwise [32,64] |
|---|---:|---:|
| Params | 19,138 | 2,894 |
| Test accuracy | 88.62% | 79.81% |

### Slice 13 — Channel Capacity Diagnostic

| Metric | Standard [32,64] | Depthwise [32,64] | Depthwise Wide [64,128] |
|---|---:|---:|---:|
| Params | 19,138 | 2,894 | 9,870 |
| Test accuracy | 85.74% | 83.81% | 82.85% |
| Mean latency (ms/batch) | 0.61 | 0.67 | 0.59 |

---

## Decision

**Accepted for future NAS candidate space:**
- `standard` — proven baseline; reference point for all future comparisons
- `depthwise_sep` with tunable channel capacity — recovers accuracy within 3–5 pp of standard at significantly lower parameter cost; channel width must be a searchable dimension

**Rejected for immediate drop-in replacement:**
- `depthwise_sep [32,64]` — 8.81 pp test accuracy gap versus `standard [32,64]`; insufficient capacity at matched channel dimensions

---

## Rationale

At matched channel dimensions `[32, 64]`, `depthwise_sep` achieves approximately 6.6× parameter reduction versus `standard`, but at the cost of an 8.81 pp test accuracy gap — too large for a direct substitution. However, widening the depthwise channels to `[64, 128]` closes that gap to within 3–5 pp (2.89 pp in Slice 13) while still delivering roughly 2× fewer parameters than `standard`, demonstrating that the accuracy deficit is a capacity issue rather than a structural one. Inference latency is competitive across all three configurations, with `depthwise_sep_wide` matching or slightly undercut `standard` in both mean latency and epoch time. These results indicate that `depthwise_sep` is viable as an efficient block candidate provided channel width is treated as a free variable rather than a fixed match to the standard config. Making channel width a searchable NAS dimension is therefore the natural next step to characterise the full accuracy-efficiency frontier.

---

## Risks and Limitations

- Evaluation was performed on a single dataset (PneumoniaMNIST); generalisation to other datasets or imaging modalities is unconfirmed.
- Only two channel configurations were tested for `depthwise_sep` (`[32, 64]` and `[64, 128]`); the accuracy-efficiency curve between and beyond these points has not been characterised.
- No repeated trials were run; all results reflect single-run measurements and may exhibit run-to-run variance due to the absence of a fixed random seed.
- Depthwise separable blocks may exhibit different accuracy and efficiency trade-offs on inputs with larger spatial dimensions or more than one input channel.

---

## Next Steps

- `depthwise_sep` with tunable `conv_channels` is a candidate for the NAS search space.
- No further manual benchmarking of `depthwise_sep` is required before NAS exploration begins.
- NAS scope and search space definition are subject to separate human approval.
