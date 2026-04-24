import argparse
import os
import datetime
import torch
import numpy as np

#internal imports
from qcore.data.medical_loader import get_medical_data
from experiments.train_cv_medmnist import train_cv_medmnist
from experiments.test_cv_medmnist import test
from experiments.plots import analyze_pca, plot_fidelity_matrix, generate_phase_diagram
from experiments.test_cv_medmnist import test, run_minimal_val
from experiments.metrics import analyze_state_separation


def main():
    parser = argparse.ArgumentParser(description="CV Medmnist Training")
    parser.add_argument("--dataset", type=str, default="pathmnist", help="MedMNIST dataset flag")
    parser.add_argument("--name", type=str, help="Model name")
    parser.add_argument("--modes", type=int, default=2, help="Number of quantum modes/PCA components")
    parser.add_argument("--depth", type=int, default=2, help="Ansatz depth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--squeeze", type=float, default=0.5)
    args = parser.parse_args()

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

    print("Generating PCA analysis")
    analyze_pca(data, run_dir=run_dir)

    #setup config dict
    config = {
        "name": args.name,
        "dataset": args.dataset,
        "n_modes": args.modes,
        "depth": args.depth,
        "epochs": args.epochs,
        "lr": args.lr,
        "noise": args.noise,
        "squeeze": args.squeeze
        # "hbar": 2.0
    }

    #train
    model, metrics = train_cv_medmnist(config, data, run_dir)

    #test
    metrics, test_results = test(model, data, run_dir)

    #test physics audit
    f_matrix = analyze_state_separation(test_results, model.n_classes, args.noise)
    plot_fidelity_matrix(f_matrix, run_dir)

    noise_range = np.arange(0.5, args.noise, 0.5)
    squeezing_range = np.arange(0.5, args.squeeze, 0.5)

    #phase diagram
    phase_results = run_minimal_val(model, data, config, run_dir, noise_range, squeezing_range)

    generate_phase_diagram(phase_results, run_dir)


if __name__ == "__main__":
    main()