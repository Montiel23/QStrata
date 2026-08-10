import argparse
import datetime
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.models.classical_resnet import ClassicalSpineResNet18
from experiments.plots import plot_inference_report_multiclass
from experiments.test_spine_dv import test_spine_dv_engine
from experiments.train_spine_dv import train_spine_classical_pipeline
# from experiments.test_spine_classical import test_spine_classical_engine
# from experiments.train_spine_classical import train_spine_classical_pipeline
from qcore.data.npy_spine_dataset import NpySpineDataset


def print_model_summary(model):
    print("\n" + "=" * 65)
    print("            CLASSICAL RESNET-18 MODEL SUMMARY")
    print("=" * 65)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"{'Component':<30} | {'Total Params':<15} | {'Trainable Params':<15}")
    print("-" * 65)
    print(
        f"{'ResNet-18 Backbone + Head':<30} | {total_params:<15,} | {trainable_params:<15,}"
    )
    print("=" * 65 + "\n")


def audit_gpu_residence(model, device):
    print("\n" + "=" * 55)
    print("            HARDWARE RESIDENCE AUDIT")
    print("=" * 55)
    print(
        f"Model Weights Device      : {next(model.parameters()).device}"
    )

    model.eval()
    dummy_x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(dummy_x)

    print(f"Output Logits on CUDA    : {logits.is_cuda} ({logits.device})")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Master Classical ResNet Spine Pipeline"
    )

    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--train_labels", type=str, required=True)
    parser.add_argument("--val_images", type=str, required=True)
    parser.add_argument("--val_labels", type=str, required=True)
    parser.add_argument("--test_images", type=str, required=True)
    parser.add_argument("--test_labels", type=str, required=True)
    parser.add_argument("--name", type=str, default="master_classical_run")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print("\n" + "=" * 60)
    print(f"Initializing QStrata Classical Master Engine on: {device}")
    print("=" * 60)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results/vindr_spine_classical/{args.name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(
        f"[Workspace] Logging all weights, metrics, and figures to: \n {run_dir}/\n"
    )

    print("[data] loading")
    train_dataset = NpySpineDataset(
        images_path=args.train_images,
        labels_path=args.train_labels,
        to_spatial_rgb=True,
    )
    val_dataset = NpySpineDataset(
        images_path=args.val_images,
        labels_path=args.val_labels,
        to_spatial_rgb=True,
    )
    test_dataset = NpySpineDataset(
        images_path=args.test_images,
        labels_path=args.test_labels,
        to_spatial_rgb=True,
    )

    total_classes = train_dataset.get_total_classes()
    class_names = train_dataset.unique_classes
    print(
        f"[Data] Discovered {total_classes} unique categories: {class_names}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch, shuffle=False
    )

    config_dict = vars(args)
    config_dict["n_classes"] = total_classes
    config_dict["class_names"] = [str(c) for c in class_names]
    config_dict["timestamp"] = timestamp

    with open(os.path.join(run_dir, "experiment_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    model = ClassicalSpineResNet18(n_classes=total_classes).to(device)

    print_model_summary(model)
    audit_gpu_residence(model, device=device)

    print("\n" + "-" * 30)
    print("optimization pipeline")
    print("-" * 30)

    model, training_history = train_spine_classical_pipeline(
        config=config_dict,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        run_dir=run_dir,
    )

    print("\n" + "-" * 30)
    print("evaluating unseen test generalization")
    print("-" * 30)

    all_labels = [labels for _, labels in test_loader]
    class_counts = torch.bincount(torch.cat(all_labels))
    weights = 1.0 / (class_counts.float() + 1e-6)
    test_criterion = nn.CrossEntropyLoss(
        weight=(weights / weights.sum()).to(device)
    )

    # test_metrics, y_true, y_logits = test_spine_classical_engine(
    test_metrics, y_true, y_logits = test_spine_dv_engine(
        model=model,
        data_loader=test_loader,
        criterion=test_criterion,
        n_classes=total_classes,
        device=device,
    )

    with open(
        os.path.join(run_dir, "test_performance_metrics.json"), "w"
    ) as f:
        json.dump(
            test_metrics,
            f,
            indent=2,
            default=lambda o: o.tolist() if hasattr(o, "tolist") else o,
        )

    print("-" * 40)
    print("generating graphics report")
    print("-" * 40)

    plots_dir = os.path.join(run_dir, "evaluation_reports")
    os.makedirs(plots_dir, exist_ok=True)

    plot_inference_report_multiclass(
        y_true=np.array(y_true),
        y_logits=np.array(y_logits),
        run_dir=plots_dir,
        n_classes=total_classes,
        class_names=class_names,
    )

    print("\n" + "=" * 50)
    print("classical experiment complete")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()