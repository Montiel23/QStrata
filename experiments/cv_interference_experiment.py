import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse

#import from your new modular structure
from qcore.ansatz.cv_ansatz import GaussianVariationalAnsatz
from experiments.models.cv_2d_classifier import CV2DClassifier
from qcore.circuit.drawer import draw_cv_ascii

def generate_2d_data(n_samples=200):
    "create 2d circular boundary dataset"
    X = torch.randn(n_samples, 2)
    # target: 1 if outside radius 1.0, else 0
    y = (torch.norm(X, dim=1) > 1.0).long()
    return X, y


def calculate_purity(cov, n_modes):
    "calculate purity of a Gaussian state: gamma = 1 / sqrt(det(V))"

    det_v = torch.det(cov)
    return 1.0 / torch.sqrt(det_v)

def train():
    parser = argparse.ArgumentParser(description="QStrata CV Training")
    parser.add_argument("--modes", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--noise", type=float, default=0.0, help="Electronic noise")
    args = parser.parse_args()

    # initialize modular framework
    ansatz = GaussianVariationalAnsatz(n_modes=args.modes, depth=args.depth)
    # model = CVGaussianClassifier(ansatz, n_classes=2)
    model = CV2DClassifier(ansatz=ansatz)
    draw_cv_ascii(model.ansatz)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    #metrics containers
    history = {"loss": [], "acc": [], "grad_norm": [], "purity": []}
    tp = fp = tn = fn = 0

    print(f"--- Training started: {args.modes} Modes, Noise: {args.noise} ---")

    for epoch in range(args.epochs):
        optimizer.zero_grad()

        #simulate small batch (XOR-like circular problem)
        X = torch.randn(32, args.modes)
        y = (torch.norm(X, dim=1) > 1.0).long()

        # logits = model(X)
        logits = model(X * 2.5)
        loss = criterion(logits, y)
        loss.backward()

        #gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        grad_norm = total_norm ** 0.5

        optimizer.step()

        #purity (checking last state in the batch)

        with torch.no_grad():
            _, last_cov = model.backend.get_vacuum()
            purity = calculate_purity(last_cov, args.modes)


        history["loss"].append(loss.item())
        history["grad_norm"].append(grad_norm)
        history["purity"].append(purity.item())

        if epoch % 10 == 0:
            # print(f"Epoch {epoch} | Acc: {}| Loss: {loss.item():.3f} | GradNorm: {grad_norm:.3e} | Purity: {purity:.4f}")
            print(f"Epoch {epoch} | Loss: {loss.item():.3f} | GradNorm: {grad_norm:.3e} | Purity: {purity:.4f}")

    # save_training_plots(history)

if __name__ == "__main__":
    train()

# def run_2d_experiment():
#     print("--- CV Experiment: 2D interference and non-linear classification ---")

#     #parameters
#     n_modes = 2
#     depth = 2
#     n_classes = 2
#     hbar = 2.0

#     ansatz = GaussianVariationalAnsatz(n_modes=n_modes, depth=depth)
#     model = CV2Dclassifier(ansatz, n_classes=n_classes, hbar=hbar)

#     #draw architecture
#     draw_cv_ascii(model.ansatz)

#     #synthetic data
#     X_train, y_train = generate_2d_data(400)
#     X_test, y_test = generate_2d_data(100)

#     #training setup
#     optimizer = optim.Adam(model.paramters(), lr=0.05)
#     criterion = nn.CrossEntropyLoss()
#     epochs = 50
#     loss = []

#     print(f"Starting training on {len(X_train)} samples")

#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()

#         #forward pass
#         logits = model(X_train)
#         loss = criterion(logits, y_train)

#         #backward pass
#         loss.backward()
#         optimizer.step()

#         loss_history.append(loss.item())

#         if epoch % 10 == 0:
            