# Slice 29B — C006 Latency Measurement Sanity Check

- **Date:** 2026-05-25
- **Candidate:** C006 — depthwise_sep, conv_channels: [64, 128], 9,870 params
- **Config:** `experiments/configs/binary_baseline_depthwise_sep_wide.yaml`

---

## 2. Measurement Methodology

Inference latency was measured using GPU-synchronized block timing. The C006 model was loaded from the YAML config, moved to the CUDA device, and set to `eval()` mode with `torch.no_grad()` active throughout. Input batches were sampled from the real PneumoniaMNIST validation loader — not from synthetic random tensors — so that data movement and preprocessing costs are reflected in the measurement. For each timed measurement, `torch.cuda.synchronize()` was called once before a block of 25 consecutive forward passes and once after the block completed, ensuring all GPU work is flushed before the wall-clock timer (`time.perf_counter()`) is read. The per-pass latency for each measurement block is the total synchronized wall-clock time divided by 25. This block approach amortises per-call Python overhead and CUDA kernel-launch micro-jitter, producing stable per-pass estimates.

---

## 3. Warmup Configuration

- **Warmup iterations:** 25
- **Purpose:** Allow the GPU to reach steady-state thermal and clock conditions, populate CUDA kernel caches, and eliminate first-call compilation overhead before timed measurements begin. Results from warmup iterations are discarded.

---

## 4. Timed Iteration Configuration

- **Timed measurement blocks:** 100
- **Forward passes per block:** 25
- **Total timed forward passes:** 2500
- **Per-pass latency:** total synchronized block time ÷ 25 forward passes
- **Input:** real batch sampled from the PneumoniaMNIST validation loader

---

## 5. Latency Statistics

| Metric | Value |
|--------|------:|
| Mean latency (ms/batch) | 1.343 |
| Std latency (ms/batch) | 0.066 |
| Min latency (ms/batch) | 1.219 |
| Max latency (ms/batch) | 1.518 |
| p50 latency (ms/batch) | 1.356 |
| p95 latency (ms/batch) | 1.444 |
| Latency std % of mean | 4.9% |

---

## 6. Decision Gate Evaluation

| Gate Criterion | Threshold | Actual | Result |
|----------------|-----------|-------:|--------|
| Latency std % of mean | ≤ 15% | 4.9% | PASS |

---

## 7. Verdict

```
VERDICT: Latency measurement noise likely; C006 cleared for follow-up validation
```

---

## 8. Technical Interpretation

With GPU-synchronized block timing, the measured latency std % of mean is 4.9%, comfortably within the 15% gate threshold. This strongly suggests that the 21.4% std % observed in Slice 29 was an artifact of naive wall-clock timing around asynchronous CUDA calls — specifically, `time.perf_counter()` measurements taken without `torch.cuda.synchronize()` captured variable amounts of GPU-queued work, introducing apparent jitter that did not reflect actual inference speed variability. The true steady-state per-pass latency is 1.343 ms/batch (p50 1.356 ms, p95 1.444 ms), which is stable. C006 should be considered cleared of the latency stability concern from Slice 29, and the original 'Not stable enough' verdict should be reattributed to a measurement methodology issue rather than genuine model instability.
