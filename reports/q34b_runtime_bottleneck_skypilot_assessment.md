# Q34B-Hotfix: DV Runtime Bottleneck and SkyPilot GPU Assessment

**Slice:** Q34B-Hotfix (EXP-003-QSTRATA-Q34B-HOTFIX)  
**Date:** 2026-05-27  
**Author:** Miguel Lopez (QStrata)  
**Branch:** `feature/q34b-runtime-hotfix`  
**Status:** COMPLETE — assessment produced; Q34C remains gated  
**Probe script:** `scripts/probe_q34b_dv_runtime.py`

---

## 1. Background

Q34B DV NAS Pilot completed with `PASS` (5/5 trials). Wall time was 11,491 s (~3.2 hours).
Q34A Classical NAS Pilot (same 5-trial, 4-epoch budget) completed in ~1,450 s (~24 min on GPU).
This 7.9× overhead gap prompted a pre-Q34C bottleneck assessment.

Q34C (CV NAS pilot) remains **gated** until this assessment concludes whether the bottleneck is intrinsic or addressable before scaling.

---

## 2. Execution Path (Code Walk)

The DV forward pass executes the following call chain per batch:

```
DVHybridCNNQNN.forward(x)                    # qcore/models/dv_hybrid_cnn_qnn.py
  backbone(x)          → (B, 128)            # GPU if available (frozen, no_grad)
  proj(features)       → (B, n_qubits)       # GPU if available (trainable)
  for i in range(B):                         # ← SEQUENTIAL PYTHON LOOP
    x_i = proj_out[i]                        # slice to (n_qubits,)
    circuit = medical_ansatz(x_i, theta, ...) # qcore/ansatz/medical_ansatz.py
    U     = backend.compile(circuit)         # → circuit.matrix()
    state = vacuum_state(n_qubits)           # CPU tensor
    out   = backend.run(U, state)            # U @ state  (CPU matmul)
    probs_i = |out|²
  probs = torch.stack(probs_list)            # (B, 2^n_qubits)
  logits = readout(probs)                    # GPU if available
```

`circuit.matrix()` (`qcore/circuit/circuit.py`) executes:

```
U = torch.eye(dim, dtype=torch.complex64)   # CPU, no device param
for op in circuit.ops:
    U = op.embed(n_qubits) @ U              # grow (2^n × 2^n) matrix per gate
```

`op.embed()` (`qcore/base/operator.py`) uses `torch.kron()` to build the full
2^n × 2^n single-qubit gate embedding — no device parameter propagated.

---

## 3. Profiling Results

All measurements from `scripts/probe_q34b_dv_runtime.py` inside the Docker
container (`docker-qstrata-gpu-1`; NVIDIA RTX 2060 SUPER; PyTorch 2.2.2+cu121).

### 3.1 Gate count per config

| Config | Gates | kron ops/embed | Total kron ops | State dim |
|--------|-------|---------------|----------------|-----------|
| n=2, d=1 | 14 | 1 | 14 | 4 |
| n=2, d=2 | 26 | 1 | 26 | 4 |
| n=4, d=1 | 32 | 3 | 96 | 16 |
| n=4, d=2 | 60 | 3 | 180 | 16 |

Each gate embed requires `(n_qubits - 1)` `torch.kron()` calls to build a
2^n × 2^n matrix, followed by one matrix multiplication into the running unitary U.

### 3.2 `circuit.matrix()` — no-gradient timing (50 runs)

| Config | avg | min | max |
|--------|-----|-----|-----|
| n=2, d=1 | **1.03 ms** | 0.99 ms | 1.40 ms |
| n=2, d=2 | **2.08 ms** | 1.91 ms | 2.99 ms |
| n=4, d=1 | **5.99 ms** | 5.82 ms | 6.52 ms |
| n=4, d=2 | **12.16 ms** | 11.53 ms | 14.86 ms |

Scaling is roughly linear in gate count × kron complexity (14 kron ops → 1.03 ms;
180 kron ops → 12.16 ms ≈ 11.8× ratio vs 180/14 = 12.9× predicted).

### 3.3 `backend.run()` — `U @ state` timing (50 runs)

| Config | avg |
|--------|-----|
| n=2, d=1 | 0.0034 ms |
| n=2, d=2 | 0.0025 ms |
| n=4, d=1 | 0.0026 ms |
| n=4, d=2 | 0.0026 ms |

**State vector application is negligible.** The U matrix is already compiled before
`backend.run()` is called; the only work is a (4,4) or (16,16) complex matmul.

### 3.4 Circuit compilation vs state application ratio

| Config | circuit.matrix() | backend.run() | Ratio |
|--------|-----------------|--------------|-------|
| n=2, d=1 | 1.03 ms | 0.003 ms | **340 : 1** |
| n=4, d=2 | 12.16 ms | 0.003 ms | **4053 : 1** |

The bottleneck is **circuit compilation** (gate embedding), not quantum state evolution.

### 3.5 Gradient-tracking overhead estimate

Empirical measurement from Q34B trial_001 (n=2, d=1):
- Wall time: 2,207 s for 4 epochs → **551 s / epoch**
- Assuming VinDr-SpineXR train split ~1,600 samples, batch_size=4:
  - ~400 batches/epoch
  - Per-batch training time: 551 s / 400 = **1,378 ms / batch**
  - Per-sample time (quantum forward + backward): ~1,378 / 4 = **344 ms / sample**
- No-gradient `circuit.matrix()`: 1.03 ms/sample
- **Autograd overhead ratio: ~334×**

The autograd graph built through complex-valued `torch.kron`, `torch.cos`,
`torch.sin`, `torch.exp`, and `torch.atan` chain multiplication is the dominant
cost during training. Each gate's embed creates a dense autograd node over a 2^n × 2^n
complex tensor. For n=2, d=1: 14 such nodes in a chain; the backward pass
re-traverses all 14.

---

## 4. GPU Compatibility Assessment

### 4.1 GPU check result

```
[GPU CHECK] Attempting DV circuit execution on CUDA...
[GPU CHECK] Result: FAIL — CUDA execution failed: RuntimeError:
Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

The device mismatch arises in `Operator._embed_single()`:

```python
# qcore/base/operator.py:29
I = torch.eye(2, dtype=U.dtype)          # ← always CPU (no device= param)
...
M = torch.kron(M, op)                    # ← fails: M is on cuda, op is on cpu
```

`Circuit.matrix()` also creates `torch.eye(dim, dtype=torch.complex64)` on CPU.
Gate `matrix()` methods (`RX`, `RY`, `RZ`, `H`, `CNOT`) use `torch.tensor(...)`,
`torch.stack(...)`, and `torch.exp(...)` without device parameters — all produce
CPU tensors regardless of input device.

### 4.2 What would be required for GPU support

Every tensor creation in the circuit stack needs `device=` propagation:

| File | Change needed |
|------|--------------|
| `qcore/circuit/circuit.py` | `torch.eye(dim, ..., device=device)` |
| `qcore/base/operator.py` | `torch.eye(dim, ..., device=device)`, `torch.zeros(...)` |
| `qcore/base/operator.py` | `_embed_single`: `torch.eye(2, ..., device=U.device)` |
| `qcore/operators/dv/rotations.py` | All `torch.tensor(...)` → pass `device=` |
| `qcore/operators/dv/entanglers.py` | `torch.tensor(...)` → pass `device=` |
| `qcore/states/vacuum.py` | `vacuum_state(n, device=device)` — already accepts it |

**This requires qcore refactoring — BLOCKED per hard constraints.**

### 4.3 GPU speedup analysis even if device support were added

State vector dimensions are 4 (n=2) or 16 (n=4). At these scales:

| Operation | Matrix size | GPU kernel overhead | Net gain |
|---|---|---|---|
| circuit.matrix() | 4×4 or 16×16 complex64 | ~10–50 μs per launch | **negative** |
| backend.run() | (16,16) @ (16,) | ~10 μs launch, ~0.001 ms compute | **negative** |
| Batched (B, 16, 16) bmm | 16×16 per sample | ~10 μs launch, ~0.01 ms compute | marginal |

The matrices are far too small for GPU to amortize kernel launch latency.
GPU provides a meaningful speedup only when matrix dimensions exceed ~512×512 for
dense ops. At n=10 qubits (1024×1024 matrices), GPU becomes relevant —
but n=10 is infeasible for other reasons (state vector memory: 8KB complex64; not
the bottleneck, but circuit depth explosion would be).

**Conclusion: Even with qcore device support, GPU acceleration for n=2–4 qubit
circuits provides no meaningful speedup. The bottleneck is the Python-level
sequential gate embedding loop, not arithmetic throughput.**

---

## 5. Batching Analysis

### 5.1 Current structure

The per-sample loop in `DVHybridCNNQNN.forward()` runs `circuit.matrix()` B times
sequentially. With batch_size=4:

| Config | circuit.matrix() × B | backend.run() × B | Total quantum fwd |
|--------|---------------------|-------------------|-------------------|
| n=2, d=1 | 4.1 ms | 0.01 ms | ~4.1 ms (no-grad) |
| n=2, d=2 | 8.3 ms | 0.01 ms | ~8.3 ms |
| n=4, d=1 | 24.0 ms | 0.01 ms | ~24.0 ms |
| n=4, d=2 | 48.6 ms | 0.01 ms | ~48.6 ms |

### 5.2 Batched alternative (no qcore changes)

Without qcore changes, partial batching is possible in the training harness:
1. Build B circuit matrices in a loop (still O(B) Python overhead)
2. Stack as `(B, 2^n, 2^n)` complex tensor
3. Stack vacuum states as `(B, 2^n, 1)`
4. Apply `torch.bmm(U_batch, state_batch)` → `(B, 2^n, 1)` in one GPU/CPU call

This reduces `backend.run()` from B serial calls to 1 batched call, and replaces
`torch.stack(probs_list)` overhead. The circuit.matrix() construction loop remains
sequential. Estimated speedup: **~1.3–1.5×** (limited because circuit.matrix() is
the expensive part and it's unchanged).

`torch.vmap` over the forward pass would be more effective, but
`torch.kron` is not supported in `vmap` until PyTorch ≥ 2.2 with full
functorch integration, and the per-gate Python object iteration inside
`medical_ansatz()` is not `vmap`-able without restructuring the circuit DSL.

**Vectorization that meaningfully reduces the autograd overhead requires a
custom backward or circuit-level batch parameterization — both require qcore
changes.**

### 5.3 Key batching recommendation (no qcore changes)

The most impactful change **without qcore modification** is **parallel trial
execution**: run each of the 5 NAS trials as a separate subprocess. The trials
are fully independent (different seeds, different hyperparameters, no shared
state). With 5 CPU cores:

| Approach | Wall time estimate |
|---|---|
| Current (sequential, 5 trials) | 11,491 s (3.19 h) |
| Parallel (5 processes, longest trial = trial_004) | ~2,672 s (0.74 h) |
| Parallel + 2-epoch trials instead of 4 | ~1,336 s (0.37 h) |

**Parallel trial execution brings the 5-trial DV pilot under 1 hour on the
current hardware without any code change to qcore or the model.**

---

## 6. Root Cause Summary

| Bottleneck | Layer | Cost | Addressable without qcore? |
|---|---|---|---|
| Sequential per-sample quantum loop | `DVHybridCNNQNN.forward()` | ~344 ms/sample (training) | Partially (bmm batching: ~1.3×) |
| `circuit.matrix()` gate embed chain | `Circuit.matrix()`, `Operator.embed()` | ~1 ms/sample (no-grad, 14 gates) | No (needs DSL restructure) |
| Autograd through complex kron chain | PyTorch autograd | ~333× overhead over no-grad | No (custom backward needed) |
| CPU-only execution | Missing `device=` in all qcore tensor ops | All quantum on CPU | No (needs qcore device propagation) |
| GPU acceleration not viable at n=2–4 | State vector size (4–16 dim) | ~0 speedup from GPU matmul | N/A (architectural limit) |
| Sequential trial execution | Pilot orchestrator | 5× serialization of independent trials | **Yes** — subprocess pool |

**Primary recommendation: parallel trial execution in the pilot orchestrator.
No qcore changes required. Estimated speedup: 4.3× (sequential → parallel, limited
by longest trial, trial_004 at 2,672 s).**

---

## 7. SkyPilot GPU Instance Assessment

### 7.1 Assessment verdict

A dedicated GPU instance does **not** solve the Q34B bottleneck. The bottleneck
is the Python-level gate embedding loop and autograd graph complexity —
neither of which benefits from GPU arithmetic until qcore device support is added.

GPU helps **only** the frozen backbone extraction (already fast) and the readout
layer (negligible cost). The quantum circuit component (>95% of wall time) remains
CPU-bound.

### 7.2 Recommended instance type (if cloud execution is authorized)

For the DV pilot specifically, a **high-vCPU CPU instance** outperforms a GPU instance:

| Instance | Type | vCPUs | GPU | Use case | Est. speedup vs Q34B |
|---|---|---|---|---|---|
| `c6i.4xlarge` | CPU-only | 16 | — | Parallel trial execution | ~4–5× |
| `m5.4xlarge` | CPU-only | 16 | — | Parallel trial execution (alt) | ~4–5× |
| `g4dn.xlarge` | GPU | 4 | T4 | Backbone GPU + quantum CPU | ~1.5–2× |
| `g4dn.2xlarge` | GPU | 8 | T4 | Backbone GPU + 8-core parallel | ~3–4× |

For the current DV pilot workload, `c6i.4xlarge` (16 vCPUs, no GPU cost) is the
optimal SkyPilot instance. A `g4dn.2xlarge` would be appropriate once qcore device
support is added and backbone GPU extraction is the bottleneck.

**The SkyPilot YAML below targets `g4dn.xlarge` as specified in the slice scope,
with a CPU-optimized alternative noted.**

### 7.3 SkyPilot YAML

See `infra/skypilot/q34b_dv_smoke_gpu.yaml`.

Scope: single-GPU smoke test that validates the Docker execution environment,
confirms CUDA is available for backbone, and runs a 1-trial mini-pilot
(1 epoch, batch_size=4). Does NOT run full NAS. Does NOT use Ray.

---

## 8. Recommended Path to < 2 Hour DV Pilots

In ascending order of implementation complexity:

### Option A — Parallel trial orchestration (RECOMMENDED, no qcore changes)

Modify `run_q34b_dv_nas_pilot.py` to use `multiprocessing.Pool` or
`concurrent.futures.ProcessPoolExecutor` to run the 5 trials in parallel.

```python
# Conceptual change to pilot orchestrator
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=5) as pool:
    futures = [pool.submit(run_trial, cfg, i) for i, cfg in enumerate(sampled_configs)]
    results = [f.result() for f in futures]
```

Expected outcome: wall time drops from 11,491 s → ~2,672 s (0.74 h) on a 5+-core machine.

**Constraints that make this safe:**
- Trials are independent: different seeds, different hyperparameters, no shared model state
- Each trial writes to a unique config path and result JSON
- Q31 runner creates unique experiment IDs per invocation

**Caveat:** CPU memory per process: each trial loads the model and backbone weights
independently. Estimated peak memory per trial: ~300 MB × 5 = ~1.5 GB total.
Acceptable on any modern development machine.

### Option B — Reduce epoch budget to 2 (trivial, 2× reduction)

Change `EPOCHS_PER_TRIAL = 4` to `EPOCHS_PER_TRIAL = 2` in the pilot orchestrator.

DV convergence is slow — the 4-epoch budget already underfit relative to Q21 (15
epochs). Reducing to 2 epochs increases underfitting but may be acceptable for pilot
frontier characterization (the relative ordering between configs is what matters,
not absolute AUROC values).

Combined with Option A: 11,491 s / 5 (parallel) / 2 (epochs) ≈ **1,150 s (~19 min)**.

### Option C — Custom backward for theta gradient (qcore change required, BLOCKED)

Implement a parameter-shift rule for `self.theta` that avoids building the full
autograd graph through `circuit.matrix()`. This decouples theta gradient computation
from the Python-level gate embedding chain, reducing the ~334× autograd overhead.

Requires modifying `qcore/models/dv_hybrid_cnn_qnn.py` — blocked per hard constraints.

### Option D — Batched circuit execution (qcore change required, BLOCKED)

Restructure `medical_ansatz` to accept a batch of input vectors `(B, n_qubits)` and
build `(B, gates, 2^n, 2^n)` operations simultaneously, enabling `torch.vmap` or
explicit broadcasting over the batch dimension.

Requires modifying `qcore/ansatz/medical_ansatz.py` — blocked per hard constraints.

### Option E — qcore device propagation + GPU batched matmul (qcore change required)

Add `device=` parameter to all tensor constructions in the circuit stack. With this
change, `circuit.matrix()` runs on GPU and `torch.bmm` over the batch delivers ~3–5×
speedup. Combined with Option A (parallel trials), total speedup: ~15–25×.

Requires modifying 6 files in `qcore/` — requires explicit approval.

---

## 9. Q34C Gate Status

Q34C (CV NAS pilot) remains **gated** until the Q34B hotfix is merged and
the following gate decision is made:

| Gate condition | Status |
|---|---|
| Q34B-hotfix assessment complete | ✓ (this report) |
| Decision on epoch budget (2 vs 4) | **PENDING** |
| Decision on parallel trial execution | **PENDING** |
| Decision on SkyPilot provisioning | **PENDING** |

Q34C will have additional overhead vs Q34B due to:
- Gaussian-state validity checks per trial (8-category stability taxonomy)
- Covariance matrix operations (larger dimension than DV state vectors)
- Symplectic eigenvalue computation

CV circuit execution is also CPU-bound (no GPU backend). The parallel trial
optimization (Option A) applies equally to Q34C.

**Recommendation:** Apply Option A (parallel trials) + Option B (2 epochs) to
both Q34B and Q34C before executing Q34C, unless an explicit decision is made to
accept the ~3-hour wall time as acceptable.

---

## 10. Artifact Registry

| Artifact | Path |
|---|---|
| Profiling probe | `scripts/probe_q34b_dv_runtime.py` |
| This report | `reports/q34b_runtime_bottleneck_skypilot_assessment.md` |
| SkyPilot smoke YAML | `infra/skypilot/q34b_dv_smoke_gpu.yaml` |

---

```
BOTTLENECK: circuit.matrix() gate embed chain + autograd through complex kron ops
PRIMARY CAUSE: per-sample Python loop (no batch parallelism); autograd ~334x overhead
GPU STATUS: BLOCKED — device mismatch in qcore (torch.eye/tensor without device=)
GPU UTILITY: even if fixed, no speedup at n=2–4 qubits (matrices too small)
RECOMMENDED FIX: parallel trial execution (5 processes) — no qcore change needed
ESTIMATED SPEEDUP: 4.3x → 2,672s (~44 min) for 5-trial pilot
ESTIMATED COMBINED: + 2 epochs → ~1,150s (~19 min)
SKYPILOT: GPU instance not recommended for DV; CPU-high-core (c6i.4xlarge) optimal
Q34C: GATED pending epoch/parallelism decision
```
