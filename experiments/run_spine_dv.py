import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

#repo imports
# from qcore.data.spine_dataset import SpineCascadeDataset
from qcore.data.npy_spine_dataset import NpySpineDataset
from experiments.models.dv_hqnn import HybridSpineDVModel
from experiments.train_spine_dv import train_spine_dv_pipeline
from experiments.test_spine_dv import test_spine_dv_engine
from experiments.metrics import audit_quantum_register, save_qubit_trajectory, compute_dv_pauli_tomography

#plotting
from experiments.plots import plot_inference_report_multiclass, plot_fidelity_matrix, plot_dv_pauli_tomography


def print_model_summary(model):
    print("\n" + "=" * 65)
    print("                    MODEL ARCHITECTURE SUMMARY")
    print("=" * 65)

    def count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    backbone_tot, backbone_trn = count_params(model.feature_extractor)
    proj_tot, proj_trn = count_params(model.quantum_projection_head)
    q_tot, q_trn = count_params(model.dv_quantum_classifier)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{'Component':<30} | {'Total Params':<15} | {'Trainable Params':<15}")
    print("-" * 65)
    print(f"{'ResNet-18 Backbone':<30} | {backbone_tot:<15,} | {backbone_trn:<15,}")
    print(f"{'Quantum Projection Head':<30} | {proj_tot:<15,} | {proj_trn:<15,}")
    print(f"{'DV Quantum Classifier':<30} | {q_tot:<15,} | {q_trn:<15,}")
    print("=" * 65)
    print(f"{'TOTAL HYBRID MODEL':<30} | {total_params:<15,} | {total_trainable:<15,}")
    print("=" * 65 + "\n")

def audit_gpu_residence(model, device):
    print("\n" + "=" * 55)
    print("      HARDWARE & QUANTUM REGISTER RESIDENCE AUDIT")
    print("=" * 55)

    q_model = model.dv_quantum_classifier

    # 1. Parameter Placement
    print(f"Backbone Weights Device     : {next(model.feature_extractor.parameters()).device}")
    print(f"Quantum weight_y Device     : {q_model.weight_y.device}")
    print(f"Quantum weight_z Device     : {q_model.weight_z.device}")
    print(f"Classical Head Device       : {q_model.classical_head.weight.device}")

    # 2. Dynamic Tensor Tracking via Dummy Pass
    model.eval()
    dummy_x = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        latent = model.feature_extractor(dummy_x)
        latent_flat = torch.flatten(latent, 1)
        q_feats = model.quantum_projection_head(latent_flat)
        psi = q_model.get_state_vector(q_feats)
        logits = model(dummy_x)

    print(f"Projection Vector on CUDA   : {q_feats.is_cuda} ({q_feats.device})")
    print(f"Quantum State |psi> on CUDA : {psi.is_cuda} ({psi.device}, {psi.dtype})")
    print(f"Final Output Logits on CUDA : {logits.is_cuda} ({logits.device})")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Master QStrata DV Spine Pipeline")

    parser.add_argument("--train_images", type=str, required=True)

    parser.add_argument("--train_labels", type=str, required=True)

    parser.add_argument("--val_images", type=str, required=True)

    parser.add_argument("--val_labels", type=str, required=True)

    parser.add_argument("--test_images", type=str, required=True)

    parser.add_argument("--test_labels", type=str, required=True)

    parser.add_argument("--name", type=str)

    parser.add_argument("--qubits", type=int, default=4)

    parser.add_argument("--depth", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--batch", type=int, default=16)

    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    # # data path arguments
    # parser.add_argument("--train_csv", type=str, required=True, help="Path to training data annotations")
    # parser.add_argument("--val_csv", type=str, required=True)
    # parser.add_argument("--test_csv", type=str, required=True)
    # parser.add_argument("--img_dir", type=str, required=True)
    # parser.add_argument("--name", type=str, default="master_dv_run")

    # #architecture hyperparameters
    # parser.add_argument("--qubits", type=int, default=4)
    # parser.add_argument("--depth", type=int, default=2)

    # #optimization hyperparameters
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--batch", type=int, default=16)
    # parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    args = parser.parse_args()

    #hardware context initialization
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print("\n" + "="*60)
    print(f"Initializing QStrata master engine on: {device}")
    print("\n" + "="*60)

    #set up dedicated timestamped log directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results/vindr_spine_dv/{args.name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Workspace] logging all weights, metrics, and figures to: \n {run_dir}/\n")

    # labels = np.load("/c/Daniel/spine-dataset/bb_data/labels-train.npy")
    # print("data type:", labels.dtype)
    # print("total unique values", len(np.unique(labels)))
    # print("first 10 values", labels[:10])
    

    # pipeline dataset loaders configuration
    print("[data] loading")
    train_dataset = NpySpineDataset(
        images_path=args.train_images,
        labels_path = args.train_labels,
        to_spatial_rgb=True,
    )

    val_dataset = NpySpineDataset(
        images_path=args.val_images,
        labels_path=args.val_labels,
        to_spatial_rgb=True
    )

    test_dataset = NpySpineDataset(
        images_path=args.test_images,
        labels_path=args.test_labels,
        to_spatial_rgb=True
    )

    total_classes = train_dataset.get_total_classes()
    class_names = train_dataset.unique_classes
    print(
        f"[Data] discovered {total_classes} unique categories: {class_names}"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    #export experiment metadata
    config_dict = vars(args)
    config_dict["n_classes"] = total_classes
    config_dict["class_names"] = [str(c) for c in class_names]
    config_dict["timestamp"] = timestamp

    with open(os.path.join(run_dir, "experiment_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    model = HybridSpineDVModel(
        n_qubits=args.qubits, depth=args.depth, n_classes=total_classes
    ).to(device)

    print_model_summary(model)
    audit_gpu_residence(model, device=device)

    #optimization loop
    print("\n" + "-" * 30)
    print("optimization pipeline")
    print("-" * 30)

    model, training_history = train_spine_dv_pipeline(
        config=config_dict,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        run_dir=run_dir
    )

    #test evaluation
    print("\n" + "-" * 30)
    print("evaluating unseen test generalization")
    print("-" * 30)

    #build inverse frequency loss class criterion for test evaluation
    all_labels = []
    for _, labels in test_loader:
        all_labels.append(labels)
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
        device=device
    )

    with open(os.path.join(run_dir, "test_performance_metrics.json"), "w") as f:
        json.dump(
            test_metrics,
            f,
            indent=2,
            default=lambda o: o.tolist() if hasattr(o, "tolist") else o,
        )

    print("-" * 40)
    print("generating graphics and quantum audit")
    print("-" * 40)

    plots_dir = os.path.join(run_dir, "evaluation_reports")
    os.makedirs(plots_dir, exist_ok=True)

    print("GRAPHICS")

    plot_inference_report_multiclass(y_true=np.array(y_true),
                                     y_logits=np.array(y_logits),
                                     run_dir=plots_dir,
                                     n_classes=total_classes,
                                     class_names=class_names)

    fidelity_matrix, test_purity = audit_quantum_register(
        model=model,
        data_loader=test_loader,
        n_qubits=args.qubits,
        n_classes=total_classes,
        device=device
    )
    print(f"test state purity: {test_purity:.4f}")

    np.save(
        os.path.join(plots_dir, "test_fidelity_matrix.npy"), fidelity_matrix
    )

    plot_fidelity_matrix(f_matrix=fidelity_matrix, run_dir=plots_dir)

    dv_data = compute_dv_pauli_tomography(model, test_loader, args.qubits, total_classes, class_names, run_dir, device)
    plot_dv_pauli_tomography(dv_data, class_names, plots_dir)

    print("\n" + "=" * 50)
    print("experiment complete")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
