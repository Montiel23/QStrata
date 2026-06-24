import argparse
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

#repo imports
from qcore.data.spine_dataset import SpineCascadeDataset
from experiments.models.dv_hqnn import HybridSpineDVModel
from experiments.train_spine_dv import train_spine_dv_pipeline
from experiments.test_spine_dv import test_spine_dv_engine
from experiments.metrics import audit_quantum_register, save_qubit_trajectory

#plotting
from experiments.plots import plot_inference_report_multiclass, plot_fidelity_matrix

def main():
    parser = argparse.ArgumentParser(description="Master QStrata DV Spine Pipeline")

    # data path arguments
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training data annotations")
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="master_dv_run")

    #architecture hyperparameters
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)

    #optimization hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
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

    # pipeline dataset loaders configuration
    print("[Data] constructing training, validation, and test partition pipelines")
    train_dataset = SpineCascadeDataset(csv_file=args.train_csv, img_dir=args.img_dir, to_spatial_rgb=True)