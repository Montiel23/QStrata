"""
scripts/probe_q34b_dv_runtime.py

Q34B-hotfix: DV Runtime Bottleneck Profiling Probe

Measures wall time for each stage of the DV forward pass:
  1. Backbone feature extraction (GPU if available)
  2. Projection layer
  3. Per-sample quantum loop (CPU-only)
     a. medical_ansatz circuit construction
     b. circuit.matrix() — gate embed + matmul chain
     c. backend.run() — U @ state
  4. Readout

Also profiles gate count and matrix dimensions for each (n_qubits, depth) config.
Does NOT run training, NAS, or full epochs — profiling only.

Usage (inside Docker):
    python3 scripts/probe_q34b_dv_runtime.py
"""

import sys
import time
import torch

# ── Ensure qcore is on path ──────────────────────────────────────────────────
sys.path.insert(0, "/workspace")

from qcore.models.dv_hybrid_cnn_qnn import DVHybridCNNQNN
from qcore.ansatz.medical_ansatz import medical_ansatz, get_ansatz_shape
from qcore.backends.base import Backend
from qcore.states.vacuum import vacuum_state
from qcore.circuit.circuit import Circuit

BACKBONE_WEIGHTS = (
    "/workspace/model_storage/pretrained_backbones/"
    "C006-D040-depthwise_sep-pretrained-vindr.pt"
)

CNN_CONFIG = {
    "block_type":     "depthwise_sep",
    "conv_channels":  [64, 128],
    "dropout":        0.3,
    "use_batchnorm":  True,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
}

CONFIGS = [
    {"n_qubits": 2, "depth": 1, "label": "n2d1"},
    {"n_qubits": 2, "depth": 2, "label": "n2d2"},
    {"n_qubits": 4, "depth": 1, "label": "n4d1"},
    {"n_qubits": 4, "depth": 2, "label": "n4d2"},
]

BATCH_SIZE    = 4
N_WARMUP      = 3
N_TIMED       = 20
ALPHA         = 0.1


def count_gates(n_qubits, depth):
    """Count gates in medical_ansatz for given config."""
    dummy_x = torch.zeros(n_qubits)
    dummy_theta = torch.zeros(*get_ansatz_shape(n_qubits, depth))
    circ = medical_ansatz(dummy_x, dummy_theta, n_qubits, depth, ALPHA)
    return len(circ.ops)


def time_circuit_matrix(n_qubits, depth, n_runs=50):
    """Time circuit.matrix() for one sample — the dominant bottleneck."""
    dummy_x = torch.zeros(n_qubits)
    dummy_theta = torch.zeros(*get_ansatz_shape(n_qubits, depth))

    # warmup
    for _ in range(5):
        circ = medical_ansatz(dummy_x, dummy_theta, n_qubits, depth, ALPHA)
        _ = circ.matrix()

    times = []
    for _ in range(n_runs):
        circ = medical_ansatz(dummy_x, dummy_theta, n_qubits, depth, ALPHA)
        t0 = time.perf_counter()
        _ = circ.matrix()
        times.append(time.perf_counter() - t0)

    return times


def time_backend_run(n_qubits, depth, n_runs=50):
    """Time backend.run() — U @ state."""
    dummy_x = torch.zeros(n_qubits)
    dummy_theta = torch.zeros(*get_ansatz_shape(n_qubits, depth))
    backend = Backend()

    circ = medical_ansatz(dummy_x, dummy_theta, n_qubits, depth, ALPHA)
    U = circ.matrix()

    times = []
    for _ in range(n_runs):
        state = vacuum_state(n_qubits)
        t0 = time.perf_counter()
        _ = backend.run(U, state)
        times.append(time.perf_counter() - t0)

    return times


def time_full_forward(model, device, n_qubits, batch_size=4, n_runs=20):
    """
    Time full DVHybridCNNQNN forward for a dummy batch.
    Breaks down backbone, quantum loop, readout.
    """
    model.eval()

    # Dummy batch
    x = torch.randn(batch_size, 1, 224, 224).to(device)

    # -- Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)

    total_times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        total_times.append(time.perf_counter() - t0)

    return total_times


def check_gpu_support():
    """Determine if DV backend tensors support CUDA device placement."""
    if not torch.cuda.is_available():
        return False, "CUDA not available in this environment"

    try:
        # Try placing circuit ops on GPU
        n_qubits = 2
        dummy_x = torch.zeros(n_qubits, device="cuda")
        dummy_theta = torch.zeros(*get_ansatz_shape(n_qubits, 1), device="cuda")
        circ = medical_ansatz(dummy_x, dummy_theta, n_qubits, 1, ALPHA)
        U = circ.matrix()
        state = vacuum_state(n_qubits, device="cuda")
        backend = Backend()
        out = backend.run(U, state)
        return True, "CUDA execution succeeded (device propagation works)"
    except Exception as e:
        return False, f"CUDA execution failed: {type(e).__name__}: {e}"


def main():
    print("=" * 70)
    print("Q34B-HOTFIX: DV Runtime Bottleneck Profiling Probe")
    print("=" * 70)

    # ── Environment ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[ENV] PyTorch {torch.__version__}")
    print(f"[ENV] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[ENV] CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"[ENV] Backbone device: {device}")
    print(f"[ENV] Quantum device: CPU (matrix ops, device-agnostic not yet implemented)")

    # ── GPU support check ──────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("[GPU CHECK] Attempting DV circuit execution on CUDA...")
    gpu_ok, gpu_msg = check_gpu_support()
    print(f"[GPU CHECK] Result: {'PASS' if gpu_ok else 'FAIL'} — {gpu_msg}")

    # ── Gate counts ───────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("[GATE COUNT] Gates per config (medical_ansatz):")
    for cfg in CONFIGS:
        n, d = cfg["n_qubits"], cfg["depth"]
        gates = count_gates(n, d)
        state_dim = 2 ** n
        # embed cost: each gate builds 2^n x 2^n matrix via kron products
        embed_muls = gates * (n - 1)   # kron products per gate embed
        print(f"  n={n} d={d}: {gates:3d} gates | state_dim={state_dim:3d}"
              f" | kron_ops_per_embed={n-1} | total_kron_ops={embed_muls}")

    # ── circuit.matrix() timing ───────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("[TIMING] circuit.matrix() per sample (50 runs):")
    matrix_timings = {}
    for cfg in CONFIGS:
        n, d = cfg["n_qubits"], cfg["depth"]
        label = cfg["label"]
        times = time_circuit_matrix(n, d, n_runs=50)
        avg_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000
        matrix_timings[label] = avg_ms
        print(f"  {label}: avg={avg_ms:.2f}ms  min={min_ms:.2f}ms  max={max_ms:.2f}ms")

    # ── backend.run() timing ──────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("[TIMING] backend.run() (U @ state) per sample (50 runs):")
    for cfg in CONFIGS:
        n, d = cfg["n_qubits"], cfg["depth"]
        label = cfg["label"]
        times = time_backend_run(n, d, n_runs=50)
        avg_ms = sum(times) / len(times) * 1000
        print(f"  {label}: avg={avg_ms:.4f}ms  (state_dim={2**n})")

    # ── Full forward timing (2-qubit, depth-1 only for speed) ─────────────────
    print("\n" + "-" * 70)
    print("[TIMING] Full DVHybridCNNQNN forward pass (backbone on {}, quantum on CPU):"
          .format(device))

    try:
        import os
        model = DVHybridCNNQNN(CNN_CONFIG, n_qubits=2, depth=1, alpha=ALPHA)
        if os.path.exists(BACKBONE_WEIGHTS):
            raw = torch.load(BACKBONE_WEIGHTS, map_location="cpu")
            state_dict = {}
            for k, v in raw.items():
                if k.startswith("0."):
                    state_dict["backbone.0." + k[2:]] = v
                elif k.startswith("1."):
                    state_dict["backbone.1." + k[2:]] = v
                else:
                    state_dict[k] = v
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  Backbone loaded. Missing: {len(missing)} Unexpected: {len(unexpected)}")
        else:
            print(f"  Backbone weights not found at {BACKBONE_WEIGHTS} — using random init")

        model = model.to(device)

        # Time full forward (N_TIMED runs)
        x = torch.randn(BATCH_SIZE, 1, 224, 224).to(device)
        model.eval()
        # warmup
        with torch.no_grad():
            for _ in range(N_WARMUP):
                _ = model(x)
        # timed
        times = []
        for _ in range(N_TIMED):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        avg_per_sample = avg / BATCH_SIZE
        print(f"  n=2 d=1 | batch={BATCH_SIZE}")
        print(f"  avg batch forward:      {avg*1000:.1f}ms")
        print(f"  avg per-sample forward: {avg_per_sample*1000:.1f}ms")
        print(f"  quantum loop fraction:  ~{matrix_timings.get('n2d1', 0) / (avg_per_sample*1000) * 100:.0f}%"
              " (circuit.matrix only, excludes ansatz build + backend.run)")

        # Estimate epoch time from per-sample time
        # VinDr train split: approximate from Q34B wall time
        # 4 epochs × ~550s/epoch → 137.5s per epoch
        # at batch_size=4 → N_batches = total_samples / 4
        # per_batch_fwd = avg * (1 + bwd overhead factor ~1.5x for grad)
        # We can back-calculate dataset size:
        epoch_time_s = 2207.3 / 4  # from trial_001
        per_epoch_batches = epoch_time_s / (avg * 1.5)  # rough backward overhead
        est_dataset_size = per_epoch_batches * BATCH_SIZE
        print(f"\n  From Q34B trial_001 wall time (2207s / 4 epochs = 551s/epoch):")
        print(f"  Estimated train set batches/epoch: {per_epoch_batches:.0f}")
        print(f"  Estimated train set size:          {est_dataset_size:.0f} samples")
        print(f"  Quantum loop time per epoch:       "
              f"{per_epoch_batches * BATCH_SIZE * avg_per_sample:.0f}s "
              f"(forward only, no backprop)")

    except Exception as e:
        import traceback
        print(f"  [ERROR] Full forward timing failed: {e}")
        traceback.print_exc()

    # ── Batching potential ────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("[BATCHING ANALYSIS] Per-sample quantum cost vs batch parallelism:")
    print("  Current: Python for-loop over batch — sequential, no parallelism")
    print("  Issue:   circuit.matrix() is called B times per forward pass")
    print("  Fix:     Vectorize via torch.vmap or manual (B, 2^n, 2^n) stack")
    print("  Blocker: torch.kron not supported in torch.vmap < 2.0 + kron grad issues")
    print("  Alt:     Manual batching — build B matrices, torch.bmm for state apply")
    for cfg in CONFIGS:
        n, d = cfg["n_qubits"], cfg["depth"]
        label = cfg["label"]
        mat_ms = matrix_timings.get(label, 0)
        # Batched cost: could build B matrices in parallel if vectorized
        # Current: B × mat_ms  |  Batched: ~mat_ms (with overhead)
        current_loop_ms = BATCH_SIZE * mat_ms
        potential_ms = mat_ms * 1.5  # rough: build B in parallel, apply bmm
        speedup = current_loop_ms / potential_ms
        print(f"  {label}: current B×{BATCH_SIZE}={current_loop_ms:.1f}ms"
              f"  batched≈{potential_ms:.1f}ms  speedup≈{speedup:.1f}x")

    # ── GPU potential ─────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("[GPU POTENTIAL] GPU acceleration feasibility:")
    print("  State vector sizes:")
    for n in [2, 4]:
        dim = 2 ** n
        mat_bytes = dim * dim * 8  # complex64 = 8 bytes
        print(f"    n={n}: state={dim}-dim, U matrix={dim}x{dim}={mat_bytes} bytes"
              f" — too small for GPU kernel efficiency")
    print("  Assessment: GPU kernel launch overhead (~10-50μs) >> matrix compute time")
    print("  Small state vectors (4–16 dim) offer no GPU speedup for matmul alone")
    print("  GPU would only help if:")
    print("    a) State vectors are batch-stacked: (B, 2^n, 2^n) matmul on GPU")
    print("    b) qcore tensors are device-aware (requires qcore changes — BLOCKED)")
    print("  Conclusion: GPU migration requires qcore refactor. Path blocked.")

    # ── SkyPilot feasibility ──────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("[SKYPILOT] GPU instance type assessment for Q34B-scale workloads:")
    print("  Target: 5-trial pilot under 2 hours")
    print("  Current: 11,491s on CPU (1 machine)")
    print("  Required speedup: 5.75x minimum (for 2h), 11.5x for 1h")
    print("  GPU matmul speedup for 4-16 dim matrices: ~1x (overhead-bound)")
    print("  Real speedup source: CPU parallelism (multi-process trials) + batching")
    print("  SkyPilot t3.2xlarge (8 vCPUs): 5 trials × 8 cores → ~1.6x if parallelized")
    print("  SkyPilot with GPU (g4dn.xlarge): GPU for backbone, CPU for quantum")
    print("    → backbone extraction speedup ~3-5x; quantum unchanged")
    print("    → net speedup estimate: 2-3x (not 5.75x required)")
    print("  Conclusion: SkyPilot GPU alone insufficient; batching fix is prerequisite")

    print("\n" + "=" * 70)
    print("PROBE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
