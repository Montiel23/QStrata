"""
scripts/benchmark_standard_vs_depthwise.py

Manual benchmark: standard block vs depthwise_sep block on PneumoniaMNIST.
Uses the two validated YAML configs unchanged. No training setting overrides.

Outputs:
  - Per-config epoch log
  - Comparison table (params, epoch time, train/val/test accuracy, latency)
  - Checkpoint decision block
"""

import os
import sys
import time

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.data.registry import get_dataset
from qcore.data.torch_adapter import TorchDatasetAdapter
from qcore.models.cnn import build_model


# ── Config paths ──────────────────────────────────────────────────────────────
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "configs")

CONFIGS = [
    ("standard",      "binary_baseline.yaml"),
    ("depthwise_sep", "binary_baseline_depthwise_sep.yaml"),
]


def load_config(filename: str) -> dict:
    path = os.path.join(CONFIGS_DIR, filename)
    with open(path) as f:
        return yaml.safe_load(f)


def compute_class_weights(dataset_adapter, num_classes: int, device: str) -> torch.Tensor:
    labels = [dataset_adapter[i][1].item() for i in range(len(dataset_adapter))]
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels_t == c).sum().float()
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


def run_benchmark(label: str, cfg_filename: str, device: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Benchmarking: {label}  ({cfg_filename})")
    print(f"{'='*60}")

    config = load_config(cfg_filename)

    # ── Hyperparameters from config (no overrides) ────────────────────────────
    epochs     = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr         = config["training"]["lr"]
    use_cw     = config["training"]["class_weights"]
    num_classes = config["dataset"]["num_classes"]

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = get_dataset(config["dataset"]["name"], "train")
    val_ds   = get_dataset(config["dataset"]["name"], "val")
    test_ds  = get_dataset(config["dataset"]["name"], "test")

    train_adapted = TorchDatasetAdapter(train_ds)
    val_adapted   = TorchDatasetAdapter(val_ds)
    test_adapted  = TorchDatasetAdapter(test_ds)

    train_loader = DataLoader(train_adapted, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_adapted,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_adapted,  batch_size=batch_size, shuffle=False)

    # ── Class weights ─────────────────────────────────────────────────────────
    if use_cw:
        cw = compute_class_weights(train_adapted, num_classes, device)
        print(f"Class weights: {cw.cpu().tolist()}")
    else:
        cw = None

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = {}
    for k, v in config["model"].items():
        model_cfg[k] = v
    for k, v in config["dataset"].items():
        model_cfg[k] = v

    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params}")

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss(weight=cw)

    epoch_times   = []
    final_train_acc = 0.0
    final_val_acc   = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        run_loss, n_cor, n_tot = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * xb.size(0)
            n_cor    += (logits.argmax(1) == yb).sum().item()
            n_tot    += xb.size(0)

        train_loss = run_loss / n_tot
        train_acc  = n_cor / n_tot

        model.eval()
        v_loss, v_cor, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                v_loss += loss_fn(logits, yb).item() * xb.size(0)
                v_cor  += (logits.argmax(1) == yb).sum().item()
                v_tot  += xb.size(0)

        val_loss = v_loss / v_tot
        val_acc  = v_cor / v_tot

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        print(
            f"  Epoch {epoch:2d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"{epoch_time:.2f}s"
        )

        final_train_acc = train_acc
        final_val_acc   = val_acc

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.eval()
    t_cor, t_tot = 0, 0
    batch_latencies = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            t0 = time.time()
            logits = model(xb)
            batch_latencies.append((time.time() - t0) * 1000)
            t_cor += (logits.argmax(1).cpu() == yb).sum().item()
            t_tot += xb.size(0)

    test_acc        = t_cor / t_tot
    mean_latency_ms = sum(batch_latencies) / len(batch_latencies)

    print(f"  Test accuracy : {test_acc * 100:.2f}%  ({t_cor}/{t_tot})")
    print(f"  Mean latency  : {mean_latency_ms:.3f} ms/batch")

    return {
        "label":            label,
        "params":           n_params,
        "mean_epoch_time":  sum(epoch_times) / len(epoch_times),
        "final_train_acc":  final_train_acc,
        "final_val_acc":    final_val_acc,
        "test_acc":         test_acc,
        "mean_latency_ms":  mean_latency_ms,
    }


def print_table(results: list) -> None:
    r0, r1 = results[0], results[1]

    def fmt_pct(v):  return f"{v * 100:.2f}%"
    def fmt_f2(v):   return f"{v:.2f}"
    def fmt_int(v):  return str(v)

    rows = [
        ("Params",                   fmt_int(r0["params"]),          fmt_int(r1["params"])),
        ("Mean epoch time (s)",      fmt_f2(r0["mean_epoch_time"]),  fmt_f2(r1["mean_epoch_time"])),
        ("Final train accuracy",     fmt_pct(r0["final_train_acc"]), fmt_pct(r1["final_train_acc"])),
        ("Final val accuracy",       fmt_pct(r0["final_val_acc"]),   fmt_pct(r1["final_val_acc"])),
        ("Test accuracy",            fmt_pct(r0["test_acc"]),        fmt_pct(r1["test_acc"])),
        ("Mean latency (ms/batch)",  fmt_f2(r0["mean_latency_ms"]),  fmt_f2(r1["mean_latency_ms"])),
    ]

    col0_w = max(len(row[0]) for row in rows) + 2
    col1_w = max(len(r0["label"]) + 2, max(len(row[1]) for row in rows) + 2, 13)
    col2_w = max(len(r1["label"]) + 2, max(len(row[2]) for row in rows) + 2, 15)

    header  = f"| {'Metric':<{col0_w}} | {r0['label']:>{col1_w}} | {r1['label']:>{col2_w}} |"
    divider = f"|{'-' * (col0_w + 2)}|{'-' * (col1_w + 2)}:|{'-' * (col2_w + 2)}:|"

    print()
    print(header)
    print(divider)
    for metric, v0, v1 in rows:
        print(f"| {metric:<{col0_w}} | {v0:>{col1_w}} | {v1:>{col2_w}} |")
    print()


def print_checkpoint_decision(results: list) -> None:
    r_std = results[0]
    r_dw  = results[1]

    acc_delta = (r_std["test_acc"] - r_dw["test_acc"]) * 100
    param_ratio = r_std["params"] / r_dw["params"]
    epoch_ratio = r_std["mean_epoch_time"] / r_dw["mean_epoch_time"]

    within_tolerance = acc_delta <= 5.0
    more_efficient   = param_ratio > 1.0 or epoch_ratio > 1.0

    print("=== Checkpoint Decision ===")
    print()
    print(
        f"1. Accuracy within 3–5 pp of standard: "
        + ("YES" if within_tolerance else "NO")
        + f" — depthwise_sep test accuracy is {r_dw['test_acc']*100:.2f}% vs "
        + f"standard {r_std['test_acc']*100:.2f}% (delta = {acc_delta:+.2f} pp)."
    )
    print(
        f"2. Efficiency improved: "
        + ("YES" if more_efficient else "NO")
        + f" — depthwise_sep has {r_dw['params']:,} params "
        + f"({param_ratio:.1f}× fewer) and mean epoch time "
        + f"{r_dw['mean_epoch_time']:.2f}s vs {r_std['mean_epoch_time']:.2f}s "
        + f"({epoch_ratio:.2f}× ratio)."
    )
    print(
        f"3. NAS candidate: "
        + ("YES" if (within_tolerance and more_efficient) else "CONDITIONAL")
        + f" — depthwise_sep should remain a NAS search space candidate "
        + ("if accuracy tolerance is acceptable for the target operating point."
           if not within_tolerance else
           "given its parameter efficiency advantage and acceptable accuracy.")
    )
    print()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", f"Expected CUDA, got: {device}. Check GPU runtime."
    print(f"Device : {device}")
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

    results = []
    for label, cfg_file in CONFIGS:
        result = run_benchmark(label, cfg_file, device)
        results.append(result)

    print_table(results)
    print_checkpoint_decision(results)


if __name__ == "__main__":
    main()
