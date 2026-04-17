import torch
from experiments.models.basic_qmodel import TwoDQClassifier
from experiments.metrics import compute_metrics, get_entropy

import torch.nn as nn
import os
import json
from tqdm import tqdm
import time
from qcore.backends.base import Backend


def train(config, data, run_dir):
    # setup configuration
    n_qubits = config["n_qubits"]
    n_classes = data["n_classes"]
    depth = config["depth"]
    alpha = config["alpha"]
    model_name = config["name"]
    epochs = config["epochs"]
    lr = config["lr"]

    # pre convert data to tensors
    X_train, y_train = data['train']
    X_val, y_val = data['val']

    x_train_input = torch.tensor(X_train, dtype=torch.float32)
    y_train_input = torch.tensor(y_train, dtype=torch.long)
    x_val_input = torch.tensor(X_val, dtype=torch.float32)
    y_val_input = torch.tensor(y_val, dtype=torch.long)

    # initialize model and optimizer
    quantum_model = TwoDQClassifier(n_qubits, depth, alpha, n_classes=n_classes)
    optimizer = torch.optim.Adam(quantum_model.parameters(), lr=lr)

    #scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    #handle class imbalance
    class_counts = torch.bincount(y_train_input)
    weights = 1.0 / (class_counts.float() + 1e-6)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)

    losses = []
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1_scores = []
    epoch_times = []


    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []

    grad_norms = []
    grad_means = []
    grad_stds = []
    state_norms = []
    entropies = []
    epoch_times = []

    lr_history = []

    circ_drawing = quantum_model.build_circuit(X_train[0])
    circ_drawing.draw()

    print("Total parameters:", quantum_model.theta.numel())

    #training loop
    for epoch in range(epochs):
        #reset confusion matrices
        train_conf_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int32)
        val_conf_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int32)
        #define epoch timer
        epoch_start_time = time.time()
        train_loop = tqdm(range(len(X_train)), desc=f"Epoch {epoch+1}/{epochs} [Train]")

        # optimizer.zero_grad()

        total_loss = torch.zeros(1)
        epoch_state_norm = 0.0
        epoch_entropy = 0.0

        tp = fp = tn = fn = 0
        val_tp = val_fp = val_tn = val_fn = 0

        # for i in range(len(X_train)):
        for i in train_loop:
            optimizer.zero_grad()


            # forward pass
            logits, out = quantum_model.forward(x_train_input[i])

            #pred, loss and backward
            pred = torch.argmax(logits).item()
            # target = int(y_train_input[i], dtype=torch.long).unsqueeze(0)
            target = y_train_input[i].view(-1)
            loss = criterion(logits.unsqueeze(0), target)
            loss.backward()

            #optimizer step for SGD
            optimizer.step()

            #loss storing
            total_loss += loss.item()

            #state norm monitoring
            norm = torch.sum(torch.abs(out)**2).real
            epoch_state_norm += norm.item()

            # train_conf_matrix[target, pred] += 1
            train_conf_matrix[int(y_train_input[i]), pred] += 1

            #entanglement entropy for every sample
            entropy = get_entropy(out, n_qubits)

            epoch_entropy += entropy.item()

        # gradient monitoring
        grad = quantum_model.theta.grad
        current_grad_norm = grad.norm().item() if grad is not None else 0.0

        # optimizer.step()

        # validation
        with torch.no_grad():

            val_loop = tqdm(range(len(X_val)), desc= f"Epoch {epoch+1}/{epochs} [Val]", leave=False)

            for j in val_loop:
                logits_val, _ = quantum_model.forward(x_val_input[j])

                pred_val = torch.argmax(logits_val).item()
                # target_val = int(y_val_input[j])

                # val_conf_matrix[target_val, pred_val] += 1
                val_conf_matrix[int(y_val_input[j]), pred_val] += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)



        tp = torch.diag(train_conf_matrix).float()
        fp = train_conf_matrix.sum(dim=0).float() - tp
        fn = train_conf_matrix.sum(dim=1).float() - tp

        total_samples = train_conf_matrix.sum().float()
        tn = total_samples - (tp + fp + fn)

        val_tp = torch.diag(val_conf_matrix).float()
        val_fp = val_conf_matrix.sum(dim=0).float() - val_tp
        val_fn = val_conf_matrix.sum(dim=1).float() - val_tp

        val_total = val_conf_matrix.sum().float()

        val_tn = val_total - (val_tp + val_fp + val_fn)


        train_acc, train_prec, train_rec, train_f1 = compute_metrics(tp, fp, tn, fn)
        val_acc, val_prec, val_rec, val_f1 = compute_metrics(val_tp, val_fp, val_tn, val_fn)

        avg_loss = total_loss / len(x_train_input)
        avg_entropy = epoch_entropy / len(x_train_input)

        epoch_duration = time.time() - epoch_start_time

        #store metrics
        # losses.append(total_loss.item())
        losses.append(avg_loss.item())
        train_accuracies.append(train_acc.mean().item())
        train_precisions.append(train_prec.mean().item())
        train_recalls.append(train_rec.mean().item())
        train_f1_scores.append(train_f1.mean().item())

        val_accuracies.append(val_acc.mean().item())
        val_precisions.append(val_prec.mean().item())
        val_recalls.append(val_rec.mean().item())
        val_f1_scores.append(val_f1.mean().item())

        entropies.append(float(avg_entropy))
        grad_norms.append(float(current_grad_norm))
        # state_norms.append(float(current_grad_norm))
        epoch_times.append(epoch_duration)


        print(
            f"Train metrics: ||"
            f"Epoch {epoch+1} |"
            # f"Loss {total_loss.item():.4f} |"
            f"Loss {avg_loss.item():.4f} |"
            f"Acc {train_acc.mean().item():.3f} |"
            f"Rec {train_rec.mean().item():.3f} |"
            f"F1 {train_f1.mean().item():.3f} ----- "
            f"Val metrics: ||"
            f"Acc {val_acc.mean().item():.3f} |"
            f"Rec {val_rec.mean().item():.3f} |"
            f"F1 {val_f1.mean().item():.3f} |"
            f"GradNorm {current_grad_norm:.3e} |"
            f"Entropy {avg_entropy:.3f}"
        )

        print(f"Time: {epoch_duration:.2f}s")

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
        "grad_norm": grad_norms,
        # "state_norm": state_norms,
        "entropy": entropies,
        "time": epoch_times,
        "learning_rates": lr_history
    }

    # save weights
    model_path = os.path.join(run_dir, model_name + ".pt")
    torch.save(quantum_model.state_dict(), model_path)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return quantum_model, metrics
