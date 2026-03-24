import argparse
import torch
import json
import os
from experiments.plots import plot_curves
from experiments.train_blobs_classifier import train
from qcore.utils import make_run_dir_from_config
# from qcore.quantum_circuit_utils import build_circuit_2q
# from experiments.two_qubit_dv_classifier import train

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_qubits", type=int, default=2)
    # parser.add_argument("--entangle", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--measure_wire", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--dataset", type=str, default="blobs")

    return parser.parse_args()


def main():
    args = parse_args()

    config = vars(args)
    # print(f"debug alpha: {config['alpha']}")

    run_dir = make_run_dir_from_config(config)

    model, metrics = train(config, run_dir)

    print(model.theta)

    #build circuit
    # circuit = build_circuit_2q(x_sample, theta, config["n_qubits"], config["depth"])
    
    # circuit.summary()
    # circuit.ascii_diagram(config["measure_wire"])

    for key, values in metrics.items():
        if isinstance(values, list):
            plot_curves(values, key, run_dir)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()