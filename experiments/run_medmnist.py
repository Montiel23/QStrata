import argparse
import torch

import datetime
import json
import os
from experiments.plots import analyze_pca
from qcore.data.medical_loader import get_medical_data
from experiments.test_medmnist import test
from experiments.train_medmnist import train

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str)
    parser.add_argument("--n_qubits", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--dataset", type=str, default="pneumonia")

    return parser.parse_args()

def run_experiment(config):

    #setup result directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"run_{timestamp}_{config['dataset']}"
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"---- Starting Experiment: {run_name} -----")

    #load data and pca
    data, pca, scaler = get_medical_data(data_flag=config['dataset'],
                                       n_components=config['n_qubits'],
                                       n_samples=config['n_samples'])
    
    print("Generating PCA analysis")
    analyze_pca(data, run_dir=run_dir)

    #training
    print("VQC training")
    trained_model, train_metrics = train(config, data, run_dir)

    #testing
    print("Evaluating on test set")
    test_metrics = test(trained_model, data, run_dir)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":


    args = parse_args()

    config = vars(args)

    # print(config)

    run_experiment(config)
