import torch
# from qcore.data.blobs import make_blobs
# from qcore.data.blobs import make_quantum_circles
# from experiments.models.blob_2q_model import Blob2QClassifier
from experiments.models.basic_qmodel import TwoDQClassifier
from experiments.metrics import compute_metrics, get_entropy
from experiments.plots import plot_boundary


import os
import json
from qcore.backends.base import Backend

from sklearn.linear_model import LogisticRegression

def train(config, run_dir):

    n_qubits = config["n_qubits"]
    depth = config["depth"]
    measure_wire = config["measure_wire"]
    alpha = config["alpha"]
    epochs = config["epochs"]
    dataset = config["dataset"]
    lr = config["lr"]

    # dataset
    if dataset == "blobs":
        from qcore.data.blobs import make_blobs
        X, y = make_blobs(200)
    elif dataset == "circles":
        from qcore.data.circles import make_quantum_circles
        X, y = make_quantum_circles(200)

    y = y.float()

    clf = LogisticRegression()
    clf.fit(X,y)
    print("Baseline accuracy:", clf.score(X,y))

    #parameter count
    n_params = depth * n_qubits
    # theta = torch.nn.Parameter(0.1 * torch.randn(n_params))

    #metrics containers
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    grad_norms = []
    grad_means = []
    grad_stds = []
    state_norms = []
    entropies = []

    backend = Backend()
    # quantum_model = Blob2QClassifier(n_qubits, depth, alpha)
    quantum_model = TwoDQClassifier(n_qubits, depth, alpha)

    optimizer = torch.optim.Adam([quantum_model.theta], lr=lr)
    
    
    circ_drawing = quantum_model.build_circuit(X[0])
    circ_drawing.draw()
    print("Total parameters:", quantum_model.theta.numel())
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        total_loss = torch.zeros(1)
        correct = 0
        epoch_state_norm = 0.0
        epoch_entropy = 0.0
        tp = fp = tn = fn = 0

        for i in range(len(X)):
            # p, out, circuit = quantum_model.forward(X[i])
            p, out = quantum_model.forward(X[i])
            
            # circuit.draw()
            
            # state norm monitoring
            norm = torch.sum(torch.abs(out)**2).real
            epoch_state_norm += norm.item()

            pred = (p > 0.5).float()

            if pred == 1 and y[i] == 1:
                tp += 1
            elif pred == 1 and y[i] == 0:
                fp += 1
            elif pred == 0 and y[i] == 0:
                tn += 1
            elif pred == 0 and y[i] == 1:
                fn += 1


            # binary crossentropy loss
            eps = 1e-6
            loss = -(y[i] * torch.log(p + eps) + (1 - y[i]) * torch.log(1 - p + eps))
            total_loss += loss

            if i == 0:
                # entanglement entropy
                entropy = get_entropy(out, n_qubits)
                
                epoch_entropy = entropy.item()

            # # entanglement entropy (2 qubits)
            # if n_qubits == 2:
            #     psi = out
            #     rho = torch.outer(psi, torch.conj(psi))

            #     rho_A = torch.zeros((2, 2), dtype=psi.dtype)

            #     for a in range(2):
            #         for b in range(2):
            #             rho_A[a, b] = (
            #                 rho[a*2 + 0, b*2 + 0] + rho[a*2 + 1, b*2 + 1]
            #             )

            #     eigvals = torch.linalg.eigvals(rho_A).real
            #     eigvals = torch.clamp(eigvals, min=1e-12)

            #     entropy = -torch.sum(eigvals * torch.log2(eigvals))

            #     epoch_entropy += entropy.item()

        #epoch aggregates
        total_loss = total_loss / len(X)
        total_loss.backward()

        # grad_norm = theta.grad.norm().item()
        grad = quantum_model.theta.grad

        if grad is not None:
            grad_norm = grad.norm().item()

        else:
            grad_norm = 0.0

        grad_norm = grad.norm().item()
        
        grad_mean = grad.mean().item()
        grad_std = grad.std().item()

        optimizer.step()

        acc, prec, rec, f1 = compute_metrics(tp, fp, tn, fn)

        epoch_state_norm /= len(X)

        if n_qubits == 2:
            epoch_entropy /= len(X)
        else:
            epoch_entropy = 0.0

        #store metrics
        losses.append(total_loss.item())
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        grad_norms.append(grad_norm)
        grad_means.append(grad_mean)
        grad_stds.append(grad_std)
        state_norms.append(epoch_state_norm)
        entropies.append(epoch_entropy)

        print(
            f"Epoch {epoch+1} |"
            f"Loss {total_loss.item():.4f} |"
            f"Acc {acc:.3f} |"
            f"Rec {rec:.3f} |"
            f"F1 {f1:.3f} |"
            f"GradNorm {grad_norm:.3e} |"
            f"Entropy {epoch_entropy:.3f}"
        )

    plot_boundary(quantum_model, X, y, "decision_boundary", run_dir)
    
    # save metrics
    metrics = {
        "loss": losses,
        "accuracy": accuracies,
        "precision": precisions,
        "recall": recalls,
        "f1": f1_scores,
        "grad_norm": state_norms,
        "entropy": entropies
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return quantum_model, metrics