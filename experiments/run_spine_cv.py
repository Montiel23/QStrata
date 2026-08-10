import argparse
import json
import os
import sys
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.metrics import (
    compute_cv_gaussian_tomography,
    compute_metrics,
    get_stats,
    compute_cv_fidelity_matrix
)

from experiments.models.cv_hqnn import HybridSpineCVModel
from experiments.plots import (
    plot_cv_gaussian_tomography,
    plot_fidelity_matrix,
    plot_inference_report_multiclass,
)

from qcore.data.npy_spine_dataset import NpySpineDataset
from experiments.train_spine_dv import train_spine_cv_pipeline
from experiments.test_spine_dv import test_spine_dv_engine


# =====================================================================
# 2. DIAGNOSTICS & SUMMARY UTILITIES
# =====================================================================
def print_cv_model_summary(model):
    print("\n" + "=" * 65)
    print("                CV HYBRID MODEL ARCHITECTURE SUMMARY")
    print("=" * 65)

    def count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        return total, trainable

    backbone_tot, backbone_trn = count_params(model.feature_extractor)
    proj_tot, proj_trn = count_params(model.quantum_projection_head)
    q_tot, q_trn = count_params(model.cv_quantum_classifier)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"{'Component':<30} | {'Total Params':<15} | {'Trainable Params':<15}")
    print("-" * 65)
    print(
        f"{'ResNet-18 Backbone':<30} | {backbone_tot:<15,} | {backbone_trn:<15,}"
    )
    print(
        f"{'Quantum Projection Head':<30} | {proj_tot:<15,} | {proj_trn:<15,}"
    )
    print(f"{'CV Symplectic Classifier':<30} | {q_tot:<15,} | {q_trn:<15,}")
    print("=" * 65)
    print(
        f"{'TOTAL HYBRID CV MODEL':<30} | {total_params:<15,} | {total_trainable:<15,}"
    )
    print("=" * 65 + "\n")


def audit_cv_gpu_residence(model, device):
    print("\n" + "=" * 55)
    print("      HARDWARE & SYMPLECTIC REGISTER RESIDENCE AUDIT")
    print("=" * 55)
    print(
        f"✓ Backbone Weights Device : {next(model.feature_extractor.parameters()).device}"
    )
    print(
        f"✓ Quantum Parameters      : {next(model.cv_quantum_classifier.parameters()).device}"
    )

    model.eval()
    dummy_x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        latent = torch.flatten(model.feature_extractor(dummy_x), 1)
        q_feats = torch.pi * model.quantum_projection_head(latent)
        r_batch, v_batch = model.cv_quantum_classifier.get_gaussian_state(
            q_feats
        )
        logits = model(dummy_x)

    print(
        f"✓ Displacement Vector r_bar: {r_batch.is_cuda} ({r_batch.device}, shape: {list(r_batch.shape)})"
    )
    print(
        f"✓ Covariance Matrix V     : {v_batch.is_cuda} ({v_batch.device}, shape: {list(v_batch.shape)})"
    )
    print(f"✓ Output Logits on CUDA    : {logits.is_cuda} ({logits.device})")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Master QStrata CV Spine Pipeline"
    )

    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--train_labels", type=str, required=True)
    parser.add_argument("--val_images", type=str, required=True)
    parser.add_argument("--val_labels", type=str, required=True)
    parser.add_argument("--test_images", type=str, required=True)
    parser.add_argument("--test_labels", type=str, required=True)
    parser.add_argument("--name", type=str, default="master_cv_run")
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hbar", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )

    args = parser.parse_args()

    # Hardware Context Initialization
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print("\n" + "=" * 60)
    print(f"Initializing QStrata CV Master Engine on: {device}")
    print("=" * 60)

    # Workspace directory allocation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results/vindr_spine_cv/{args.name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(
        f"[Workspace] Logging all weights, metrics, and figures to: \n {run_dir}/\n"
    )

    # Pipeline Dataset Loaders Configuration
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

    # Export Experiment Configuration JSON
    config_dict = vars(args)
    config_dict["n_classes"] = total_classes
    config_dict["class_names"] = [str(c) for c in class_names]
    config_dict["timestamp"] = timestamp

    with open(os.path.join(run_dir, "experiment_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Model Instantiation
    model = HybridSpineCVModel(
        n_modes=args.modes,
        depth=args.depth,
        n_classes=total_classes,
        hbar=args.hbar,
    ).to(device)

    print_cv_model_summary(model)
    audit_cv_gpu_residence(model, device=device)

    # Optimization Loop
    print("\n" + "-" * 30)
    print("optimization pipeline")
    print("-" * 30)

    model, training_history = train_spine_cv_pipeline(
        config=config_dict,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        run_dir=run_dir,
        hbar=args.hbar,
        alpha=args.alpha,
    )

    # Test Partition Evaluation
    print("\n" + "-" * 30)
    print("evaluating unseen test generalization")
    print("-" * 30)

    # Build Class Weighting Criterion for Evaluation
    all_labels = [labels for _, labels in test_loader]
    class_counts = torch.bincount(torch.cat(all_labels))
    weights = 1.0 / (class_counts.float() + 1e-6)
    test_criterion = nn.CrossEntropyLoss(
        weight=(weights / weights.sum()).to(device)
    )

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

    # Graphics Generation & Symplectic Quantum Audit
    print("-" * 40)
    print("generating graphics and quantum audit")
    print("-" * 40)

    plots_dir = os.path.join(run_dir, "evaluation_reports")
    os.makedirs(plots_dir, exist_ok=True)

    print("GRAPHICS")

    # 1. Multi-class ROC, Precision-Recall & Confusion Matrix
    plot_inference_report_multiclass(
        y_true=np.array(y_true),
        y_logits=np.array(y_logits),
        run_dir=plots_dir,
        n_classes=total_classes,
        class_names=class_names,
    )

    # 2. Phase-Space Gaussian State Tomography (<q>, <p>, <n>, purity)
    cv_tomography_data = compute_cv_gaussian_tomography(
        model=model,
        data_loader=test_loader,
        n_modes=args.modes,
        n_classes=total_classes,
        device=device,
        hbar=args.hbar,
    )
    plot_cv_gaussian_tomography(
        tomography_data=cv_tomography_data,
        class_names=class_names,
        run_dir=plots_dir,
    )

    # 3. Exact Bures Gaussian Fidelity Overlap Matrix
    fidelity_matrix = compute_cv_fidelity_matrix(
        cv_tomography_data=cv_tomography_data,
        n_classes=total_classes,
        hbar=args.hbar,
    )
    np.save(
        os.path.join(plots_dir, "test_fidelity_matrix.npy"), fidelity_matrix
    )
    plot_fidelity_matrix(f_matrix=fidelity_matrix, run_dir=plots_dir)

    print("\n" + "=" * 50)
    print("experiment complete")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

