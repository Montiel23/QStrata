import argparse
import os
import datetime
import torch
import numpy as np

#internal imports
from qcore.data.medical_loader import get_medical_data
from experiments.train_cv_medmnist import train_cv_medmnist
from experiments.plots import analyze_pca
from experiments.test_cv_medmnist import test

def main():
    parser = argparse.ArgumentParser(description="CV Medmnist Training")
    parser.add_argument("--dataset", type=str, default="pathmnist", help="MedMNIST dataset flag")
    parser.add_argument("--name", type=str, help="Model name")
    parser.add_argument("--modes", type=int, default=2, help="Number of quantum modes/PCA components")
    parser.add_argument("--depth", type=int, default=2, help="Ansatz depth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=int, default=0.01)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.parse_args()

    #unique results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results/{args.dataset}/{args.name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    #load and encode data via PCA
    print(f"Loading {args.dataset} with {args.modes} PCA components")
    data, pca, scaler = get_medical_data(
        data_flag=args.dataset,
        n_components=args.modes,
        n_samples=args.samples
    )

    #setup config dict
    config = {
        "name": args.name,
        "dataset": args.dataset,
        "n_modes": args.modes,
        "depth": args.depth,
        "epochs": args.epochs,
        "lr": args.lr,
        "noise": args.noise,
        "hbar": 2.0
    }

    #train
    model, metrics = train_cv_medmnist(config, data, run_dir)

    #test
    metrics = test_cv_medmnist(model, data, run_dir)

if __name__ == "__main__":
    main()