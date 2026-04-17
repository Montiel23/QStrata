import torch
import torch.nn as nn
import os
import json
import time
from tqdm import tqdm
import numpy as np

from qcore.ansatz.cv_ansatz import GaussianVariationalAnsatz
from experiments.models.cv_2d_classifier import CV2DClassifier
from experiments.metrics import compute_metrics, calculate_purity
from circuit.cv_drawing import draw_cv_ascii

def train_cv_medmnist(config, data, run_dir):
    # setup configuration
    n_modes = config["n_modes"]
    n_classes = data["n_classes"]
    depth = config["depth"]
    epochs = config["epochs"]
    lr = config["lr"]
    hbar = config.get("hbar", 2.0)

    X_train, y_train = data["train"]

    x_train_input = torch.tensor(X_train, dtype=torch.float32)
    y_train_input = torch.tensor(y_train, dtype=torch.long)


    
    X_val, y_val = data["val"]

    #initialize model
    ansatz = GaussianVariationalAnsatz(n_modes=n_modes, depth=depth)
    model = CV2DClassifier(ansatz, n_classes=n_classes, hbar=hbar)

    #draw circuit once for verification
    draw_cv_ascii(model.ansatz)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    #handle class imbalance in MedMNIST
    class_counts = torch.bincount(torch.tensor(y_train, dtype=torch.long))
    weights = 1.0 / (class_counts.float() + 1e-6)
    criterion = nn.CrossEntropyLoss(weight=weights / weights.sum())

    #metric containers
    history = {
        "train_loss": [], "train_acc": [], "train_prec": [], "train_rec": [], "train_f1": [], 
        "val_acc": [], "val_prec": [], "val_rec": [], "val_f1": [],
        "grad_norm": [], "purity": [], "time": [], "learning_rates": []
    }

    lr_history = []

    for epoch in range(epochs):
        epoch_start = time.time()
        train_conf_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int32)
        total_loss = 0.0
        epoch_purity = 0.0

        # train_loop = tqdm(range(len(X_train)), desc=f"Epoch {epoch+1}/{epochs} [CV Train]")
        train_loop = tqdm(range(len(x_train_input)), desc=f"Epoch {epoch+1}/{epochs} [CV Train]")

        model.train()
        for i in train_loop:
            optimizer.zero_grad()

            #forward pass
            # x_input = torch.tensor(X_train[i], dtype=torch.float32).unsqueeze(0)
            # target = torch.tensor([y_train[i]], dtype=torch.long)

            # logits = model(x_input)
            # loss = criterion(logits, target)

            logits = model(x_train_input[i])
            loss = criterion(logits, y_train[i])

            #backward pass
            loss.backward()

            #physics monitoring: grad norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            current_grad_norm = total_norm ** 0.5

            

            optimizer.step()

            #metrics
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1).item()
            train_conf_matrix[target.item(), pred] += 1

            #physics monitoring: purity 
            _, last_conv = model.backend.get_vacuum()
            epoch_purity += calculate_purity(last_cov, hbar)

        # validation phase
        model.eval()
        val_conf_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int32)
        with torch.no_grad():
            for j in range(len(X_val)):
                x_v = torch.tensor(X_val[j], dtype=torch.float32).unsqueeze(0)
                l_v = model(x_v)
                val_conf_matrix[int(y_val[j]), torch.argmax(l_v).item()] += 1

        #epoch metrics
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)
        
        avg_loss = total_loss / len(X_train)

        #metric calculation
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

        avg_loss = total_loss / len(X_train)
        grad_norm

        epoch_duration = time.time() - epoch_start_time

        #store metrics

        history["train_loss"].append(avg_loss.item())
        history["train_acc"].append(train_acc.mean().item())
        history["train_rec"].append(train_rec.mean().item())
        history["train_prec"].append(train_prec.mean().item())
        history["train_f1"].append(train_f1.mean().item())
        
        history["val_acc"].append(val_acc.mean().item())
        history["val_rec"].append(val_rec.mean().item())
        history["val_prec"].append(val_prec.mean().item())
        history["val_f1"].append(val_f1.mean().item())

        history["grad_norm"].append(float(
        history["purity"].append(float(epoch_purity))
        history["epoch_time"].append(epoch_duration)

        print(
            f"Train metrics: ||"
            f"Epoch {epoch+1} |"
            f"Loss {avg_loss.item():.4f} |"
            f"Acc {train_acc.mean().item():.3f} |"
            f"Rec {train_rec.mean().item():.3f} |"
            f"F1 {train_f1.mean().item():3f} ---- "
            f"Val metrics: ||"
            f"Acc {val_acc.mean().item():.3f} |"
            f"Rec {val_rec.mean().item():.3f} |"
            f"F1 {val_f1.mean().item():.3f} |"
            f"GradNorm {current_grad_norm:.3e} |"
            f"Entropy {avg_entropy:.3f}"
        )

        print(f"Time: {epoch_duration:.2f}s")


    metrics = {
        "train_loss": history["train_loss"],
        "train_accuracy": history["train_acc"],
        "train_recall": history["train_rec"],
        "train_precision": history["train_prec"],
        "train_f1": history["train_f1"],
        "val_accuracy": history["val_acc"],
        "val_precision": history["val_rec"],
        "val_recall": history["val_prec"],
        "val_f1": history["val_f1"],
        "grad_norm": history["grad_norm"],
        "time": history["epoch_time"],
        "purity": history["purity"],
        "learning_rates": lr_history
    }

    #save weights
    model_path = os.path.join(run_dir, model_name + ".pt")
    torch.save(quantum_model.state_dict(), model_path)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dum(config, f, indent=2)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return quantum_model, metrics