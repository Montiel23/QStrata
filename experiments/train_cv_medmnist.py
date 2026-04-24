import torch
import torch.nn as nn
import os
import json
import time
from tqdm import tqdm
import numpy as np

from qcore.ansatz.cv_ansatz import GaussianVariationalAnsatz
from experiments.models.cv_2d_classifier import CV2DClassifier
from experiments.metrics import compute_metrics, calculate_purity, count_quantum_resources
from qcore.circuit.drawer import draw_cv_ascii

from experiments.plots import plot_mode_wigner


def train_cv_medmnist(config, data, run_dir):
    # setup configuration
    n_modes = config["n_modes"]
    model_name = config["name"]
    n_classes = data["n_classes"]
    depth = config["depth"]
    epochs = config["epochs"]
    lr = config["lr"]
    # hbar = config.get("hbar", 2.0)
    hbar = config["noise"]

    X_train, y_train = data["train"]

    x_train_input = torch.tensor(X_train, dtype=torch.float32)
    y_train_input = torch.tensor(y_train, dtype=torch.long)
    
    X_val, y_val = data["val"]

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    #initialize model
    ansatz = GaussianVariationalAnsatz(n_modes=n_modes, depth=depth)
    resources = count_quantum_resources(ansatz=ansatz)

    print(f"\nTrainable weights {resources['Trainable_weights']} |"
          f"Single-Mode gates {resources['Single_mode_gates']} |"
          f"Two-Mode Gates {resources['Two_mode_gates']} |\n")
    
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
        "grad_norm": [], "purity": [], "epoch_time": [], "learning_rates": []
    }

    lr_history = []

    for epoch in range(epochs):
        epoch_start = time.time()
        train_conf_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int32)
        total_loss = 0.0
        epoch_purity = 0.0
        epoch_grad_norms = []

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

            # logits = model(x_train_input[i])
            # loss = criterion(logits, y_train_input[i])
            logits = model(x_train_input[i].unsqueeze(0))
            target = y_train_input[i].unsqueeze(0)
            loss = criterion(logits, target)

            #backward pass
            loss.backward()

            #physics monitoring: grad norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            current_grad_norm = total_norm ** 0.5

            epoch_grad_norms.append(current_grad_norm)


            #gradient clipping preventing exploding gradient
            # torch.nn.utils.clip_grad_norm_(model.parameters(), mar_norm=1.0)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "squeezing" in name:
                        # param.clamp_(-2.0, 2.0)
                        limit = config.get("max_squeezing", 2.0)
                        param.clamp_(-limit, limit)

            #metrics
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1).item()
            train_conf_matrix[target.item(), pred] += 1

            #physics monitoring: purity 
            _, last_conv = model.backend.get_vacuum()
            epoch_purity += calculate_purity(last_conv, hbar)


        avg_grad_norm = np.mean(epoch_grad_norms)
        history["grad_norm"].append(float(avg_grad_norm))
        
        
        # validation phase
        model.eval()
        val_conf_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int32)
        with torch.no_grad():
            for j in range(len(X_val)):
                logits = model(X_val[j].unsqueeze(0))


                pred = torch.argmax(logits, dim=1).item()
                target_idx = y_val[j].item()

                val_conf_matrix[target_idx, pred] += 1

                
                # target = y_val_input[j].unsqueeze(0)
                # x_v = torch.tensor(X_val[j], dtype=torch.float32).unsqueeze(0)
                # l_v = model(x_v)
                # val_conf_matrix[int(y_val[j]), torch.argmax(l_v).item()] += 1


        if epoch % 1 == 0:
            # use a consistent sample to see how its Wigner state evolves
            sample_mu, sample_cov = model.get_state_for_sample(X_val[0])
            plot_mode_wigner(sample_mu, sample_cov, mode_idx=0, save_path = os.path.join(run_dir, f"trace_epoch_{epoch}.png"))
        
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
        # grad_norm

        epoch_duration = time.time() - epoch_start

        #store metrics

        # history["train_loss"].append(avg_loss.item())
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc.mean().item())
        history["train_rec"].append(train_rec.mean().item())
        history["train_prec"].append(train_prec.mean().item())
        history["train_f1"].append(train_f1.mean().item())
        
        history["val_acc"].append(val_acc.mean().item())
        history["val_rec"].append(val_rec.mean().item())
        history["val_prec"].append(val_prec.mean().item())
        history["val_f1"].append(val_f1.mean().item())

        # history["grad_norm"].append(float(
        history["purity"].append(float(epoch_purity) / len(x_train_input))
        history["epoch_time"].append(epoch_duration)

        print(
            f"Train metrics: ||"
            f"Epoch {epoch+1} |"
            # f"Loss {avg_loss.item():.4f} |"
            f"Loss {avg_loss:.4f} |"
            f"Acc {train_acc.mean().item():.3f} |"
            f"Rec {train_rec.mean().item():.3f} |"
            f"F1 {train_f1.mean().item():.3f} ---- "
            f"Val metrics: ||"
            f"Acc {val_acc.mean().item():.3f} |"
            f"Rec {val_rec.mean().item():.3f} |"
            f"F1 {val_f1.mean().item():.3f} |"
            # f"GradNorm {current_grad_norm:.3e} |"
            f"GradNorm {avg_grad_norm:.3e} |"
            f"Purity {epoch_purity / len(x_train_input):.3f} |"
            # f"Entropy {avg_entropy:.3f}"
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
    # torch.save(quantum_model.state_dict(), model_path)
    torch.save(model.state_dict(), model_path)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # return quantum_model, metrics
    return model, metrics