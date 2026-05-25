"""
scripts/audit_nas_v0_benchmark.py

Slice 19 audit: re-run C001 (binary_baseline.yaml) only.
Isolated from all previous benchmark scripts.
Prints full per-epoch log and collected metrics.
No training setting overrides; YAML config is the sole authority.
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


REPO_ROOT   = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(REPO_ROOT, "experiments", "configs", "binary_baseline.yaml")


def compute_class_weights(dataset_adapter, num_classes: int, device: str) -> torch.Tensor:
    labels = [dataset_adapter[i][1].item() for i in range(len(dataset_adapter))]
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels_t == c).sum().float()
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


def run_c001_audit(device: str) -> dict:
    print(f"\n{'='*60}")
    print(f"AUDIT RE-RUN: C001  (binary_baseline.yaml)")
    print(f"{'='*60}")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    print("Config loaded:")
    print(f"  model:    {config['model']}")
    print(f"  training: {config['training']}")
    print(f"  dataset:  {config['dataset']}")

    # ── Hyperparameters from config — no overrides ────────────────────────────
    epochs      = config["training"]["epochs"]
    batch_size  = config["training"]["batch_size"]
    lr          = config["training"]["lr"]
    use_cw      = config["training"]["class_weights"]
    num_classes = config["dataset"]["num_classes"]

    # ── Data — verify each split is loaded explicitly ─────────────────────────
    print("\nLoading dataset splits...")
    train_ds = get_dataset(config["dataset"]["name"], "train")
    val_ds   = get_dataset(config["dataset"]["name"], "val")
    test_ds  = get_dataset(config["dataset"]["name"], "test")

    train_adapted = TorchDatasetAdapter(train_ds)
    val_adapted   = TorchDatasetAdapter(val_ds)
    test_adapted  = TorchDatasetAdapter(test_ds)

    print(f"  train split size : {len(train_adapted)}")
    print(f"  val split size   : {len(val_adapted)}")
    print(f"  test split size  : {len(test_adapted)}")

    train_loader = DataLoader(train_adapted, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_adapted,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_adapted,  batch_size=batch_size, shuffle=False)

    # ── Verify label distributions (quick sanity check) ───────────────────────
    for split_name, adapter in [("train", train_adapted), ("val", val_adapted), ("test", test_adapted)]:
        labels = [adapter[i][1].item() for i in range(len(adapter))]
        n0 = labels.count(0)
        n1 = labels.count(1)
        print(f"  {split_name:5s} label dist: class0={n0}, class1={n1}")

    # ── Class weights ─────────────────────────────────────────────────────────
    if use_cw:
        cw = compute_class_weights(train_adapted, num_classes, device)
        print(f"\nClass weights: {cw.cpu().tolist()}")
    else:
        cw = None

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = {}
    for k, v in config["model"].items():
        model_cfg[k] = v
    for k, v in config["dataset"].items():
        model_cfg[k] = v

    print(f"\nBuilding model with config: {model_cfg}")
    model    = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Architecture:\n{model}")

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss(weight=cw)

    print(f"\nTraining for {epochs} epochs (lr={lr}, batch_size={batch_size})...")

    epoch_times     = []
    final_train_acc = 0.0
    final_val_acc   = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
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

        # ── Validate ──────────────────────────────────────────────────────────
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
    t_cor, t_tot    = 0, 0
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

    print(f"\n--- Final metrics ---")
    print(f"  Params            : {n_params:,}")
    print(f"  Final train acc   : {final_train_acc * 100:.2f}%")
    print(f"  Final val acc     : {final_val_acc * 100:.2f}%")
    print(f"  Test accuracy     : {test_acc * 100:.2f}%  ({t_cor}/{t_tot})")
    print(f"  Mean latency      : {mean_latency_ms:.3f} ms/batch")

    return {
        "params":            n_params,
        "final_train_acc":   final_train_acc,
        "final_val_acc":     final_val_acc,
        "test_acc":          test_acc,
        "mean_latency_ms":   mean_latency_ms,
        "mean_epoch_time":   sum(epoch_times) / len(epoch_times),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", f"Expected CUDA, got: {device}. Check GPU runtime."
    print(f"Device : {device}")
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

    metrics = run_c001_audit(device)

    print("\n" + "=" * 60)
    print("AUDIT RE-RUN COMPLETE")
    print("=" * 60)
    print(f"  Params            : {metrics['params']:,}")
    print(f"  Mean epoch time   : {metrics['mean_epoch_time']:.2f}s")
    print(f"  Final train acc   : {metrics['final_train_acc'] * 100:.2f}%")
    print(f"  Final val acc     : {metrics['final_val_acc'] * 100:.2f}%")
    print(f"  Test accuracy     : {metrics['test_acc'] * 100:.2f}%")
    print(f"  Mean latency      : {metrics['mean_latency_ms']:.3f} ms/batch")


if __name__ == "__main__":
    main()
