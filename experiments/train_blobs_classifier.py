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
from sklearn.model_selection import train_test_split

def train(config, run_dir):

    n_qubits = config["n_qubits"]
    depth = config["depth"]
    measure_wire = config["measure_wire"]
    alpha = config["alpha"]
    epochs = config["epochs"]
    dataset = config["dataset"]
    lr = config["lr"]
    model_name = config["name"]

    # dataset
    if dataset == "blobs":
        from qcore.data.blobs import make_blobs
        X, y = make_blobs(200)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    elif dataset == "circles":
        from qcore.data.circles import make_quantum_circles
        X, y = make_quantum_circles(200)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    y_train, y_val = y_train.float(), y_val.float()
    # y = y.float()

    clf = LogisticRegression()
    # clf.fit(X,y)
    clf.fit(X_train, y_train)
    # print("Baseline accuracy:", clf.score(X,y))
    print("Baseline accuracy:", clf.score(X_train, y_train))

    #parameter count
    n_params = depth * n_qubits
    # theta = torch.nn.Parameter(0.1 * torch.randn(n_params))

    #metrics containers
    losses = []
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1_scores = []
    
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []

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
    
    #training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        total_loss = torch.zeros(1)
        correct = 0
        epoch_state_norm = 0.0
        epoch_entropy = 0.0
        tp = fp = tn = fn = 0
        val_tp = val_fp = val_tn = val_fn = 0

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


        # validation
        with torch.no_grad():

            for j in range(len(X_val)):
                p_val, _ = quantum_model.forward(X_val[j])
                val_pred = (p_val > 0.5).float()
                if val_pred == 1 and y_val[j] == 1:
                    val_tp += 1
                elif val_pred == 1 and y_val[j] == 0:
                    val_fp += 1
                elif val_pred == 0 and y_val[j] == 0:
                    val_tn += 1
                elif val_pred == 0 and y_val[j] == 1:
                    val_fn += 1
        
        train_acc, train_prec, train_rec, train_f1 = compute_metrics(tp, fp, tn, fn)
        val_acc, val_prec, val_rec, val_f1 = compute_metrics(val_tp, val_fp, val_tn, val_fn)

        epoch_state_norm /= len(X)

        if n_qubits == 2:
            epoch_entropy /= len(X)
        else:
            epoch_entropy = 0.0

        #store metrics
        losses.append(total_loss.item())
        train_accuracies.append(train_acc)
        train_precisions.append(train_prec)
        train_recalls.append(train_rec)
        train_f1_scores.append(train_f1)

        val_accuracies.append(val_acc)
        val_precisions.append(val_prec)
        val_recalls.append(val_rec)
        val_f1_scores.append(val_f1)

        grad_norms.append(grad_norm)
        grad_means.append(grad_mean)
        grad_stds.append(grad_std)
        state_norms.append(epoch_state_norm)
        entropies.append(epoch_entropy)


        

        print(
            f"Train metrics: ||"
            f"Epoch {epoch+1} |"
            f"Loss {total_loss.item():.4f} |"
            f"Acc {train_acc:.3f} |"
            f"Rec {train_rec:.3f} |"
            f"F1 {train_f1:.3f} ----- "
            f"Val metrics: ||"
            f"Acc {val_acc:.3f} |"
            f"Rec {val_rec:.3f} |"
            f"F1 {val_f1:.3f} |"
            f"GradNorm {grad_norm:.3e} |"
            f"Entropy {epoch_entropy:.3f}"
        )

    plot_boundary(quantum_model, X, y, "decision_boundary", run_dir)
    
    # save metrics
    metrics = {
        "train_loss": losses,
        "train_accuracy": train_accuracies,
        "train_precision": train_precisions,
        "train_recall": train_recalls,
        "train_f1": train_f1_scores,
        "val_accuracy": val_accuracies,
        "val_precision": val_precisions,
        "val_recall": val_recalls,
        "val_f1": val_f1_scores,
        "grad_norm": state_norms,
        "entropy": entropies
    }

    #save weights
    model_path = os.path.join(run_dir, model_name + ".pt")
    torch.save(quantum_model.state_dict(), model_path)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return quantum_model, metrics, (X_test, y_test)