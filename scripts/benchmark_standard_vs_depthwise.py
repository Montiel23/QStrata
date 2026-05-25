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
    ("standard",           "binary_baseline.yaml"),
    ("depthwise_sep",      "binary_baseline_depthwise_sep.yaml"),
    ("depthwise_sep_wide", "binary_baseline_depthwise_sep_wide.yaml"),
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
    r0, r1, r2 = results[0], results[1], results[2]

    def fmt_pct(v):  return f"{v * 100:.2f}%"
    def fmt_f2(v):   return f"{v:.2f}"
    def fmt_int(v):  return str(v)

    rows = [
        ("Params",                   fmt_int(r0["params"]),          fmt_int(r1["params"]),          fmt_int(r2["params"])),
        ("Mean epoch time (s)",      fmt_f2(r0["mean_epoch_time"]),  fmt_f2(r1["mean_epoch_time"]),  fmt_f2(r2["mean_epoch_time"])),
        ("Final train accuracy",     fmt_pct(r0["final_train_acc"]), fmt_pct(r1["final_train_acc"]), fmt_pct(r2["final_train_acc"])),
        ("Final val accuracy",       fmt_pct(r0["final_val_acc"]),   fmt_pct(r1["final_val_acc"]),   fmt_pct(r2["final_val_acc"])),
        ("Test accuracy",            fmt_pct(r0["test_acc"]),        fmt_pct(r1["test_acc"]),        fmt_pct(r2["test_acc"])),
        ("Mean latency (ms/batch)",  fmt_f2(r0["mean_latency_ms"]),  fmt_f2(r1["mean_latency_ms"]),  fmt_f2(r2["mean_latency_ms"])),
    ]

    col0_w = max(len(row[0]) for row in rows) + 2
    col1_w = max(len(r0["label"]) + 2, max(len(row[1]) for row in rows) + 2, 13)
    col2_w = max(len(r1["label"]) + 2, max(len(row[2]) for row in rows) + 2, 15)
    col3_w = max(len(r2["label"]) + 2, max(len(row[3]) for row in rows) + 2, 20)

    header  = (f"| {'Metric':<{col0_w}} | {r0['label']:>{col1_w}} | "
               f"{r1['label']:>{col2_w}} | {r2['label']:>{col3_w}} |")
    divider = (f"|{'-' * (col0_w + 2)}|{'-' * (col1_w + 2)}:|"
               f"{'-' * (col2_w + 2)}:|{'-' * (col3_w + 2)}:|")

    print()
    print(header)
    print(divider)
    for metric, v0, v1, v2 in rows:
        print(f"| {metric:<{col0_w}} | {v0:>{col1_w}} | {v1:>{col2_w}} | {v2:>{col3_w}} |")
    print()


def print_checkpoint_decision(results: list) -> None:
    r_std  = results[0]
    r_wide = results[2]

    acc_delta   = (r_std["test_acc"] - r_wide["test_acc"]) * 100
    param_ratio = r_std["params"] / r_wide["params"]
    lat_ratio   = r_wide["mean_latency_ms"] / r_std["mean_latency_ms"]
    epoch_ratio = r_wide["mean_epoch_time"] / r_std["mean_epoch_time"]

    recovered        = acc_delta <= 5.0
    param_efficient  = param_ratio > 1.0
    latency_compet   = lat_ratio < 2.0 and epoch_ratio < 2.0
    nas_candidate    = recovered and param_efficient

    print("=== Checkpoint Decision ===")
    print()
    print(
        f"1. Wider depthwise recovered test accuracy within 3–5 pp of standard: "
        + ("YES" if recovered else "NO")
        + f" — depthwise_sep_wide test accuracy is {r_wide['test_acc']*100:.2f}% vs "
        + f"standard {r_std['test_acc']*100:.2f}% (delta = {acc_delta:+.2f} pp)."
    )
    print(
        f"2. Wider depthwise remained meaningfully more parameter-efficient than standard: "
        + ("YES" if param_efficient else "NO")
        + f" — depthwise_sep_wide has {r_wide['params']:,} params vs "
        + f"standard {r_std['params']:,} ({param_ratio:.2f}× fewer)."
    )
    print(
        f"3. Wider depthwise remained competitive in latency and epoch time: "
        + ("YES" if latency_compet else "NO")
        + f" — mean latency {r_wide['mean_latency_ms']:.2f} ms/batch vs "
        + f"standard {r_std['mean_latency_ms']:.2f} ms/batch ({lat_ratio:.2f}×); "
        + f"mean epoch time {r_wide['mean_epoch_time']:.2f}s vs "
        + f"{r_std['mean_epoch_time']:.2f}s ({epoch_ratio:.2f}×)."
    )
    print(
        f"4. depthwise_sep should remain in the future NAS candidate space: "
        + ("YES" if nas_candidate else "CONDITIONAL")
        + (" — wider channels recovered accuracy within tolerance while retaining "
           "parameter efficiency, making depthwise_sep a valid NAS candidate."
           if nas_candidate else
           " — accuracy gap versus standard exceeds 5 pp even at wider channels; "
           "depthwise_sep may need further investigation before inclusion in NAS.")
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
