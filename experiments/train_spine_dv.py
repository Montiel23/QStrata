import torch
import torch.nn as nn
from tqdm import tqdm
import os
import time
import json
from tqdm import tqdm
import numpy as np

from experiments.metrics import compute_metrics, save_qubit_trajectory, get_stats

def train_spine_dv_pipeline(config, train_loader, val_loader, model, device, run_dir):

    model_name = config["name"]
    n_classes = config["n_classes"]
    epochs = config["epochs"]
    lr = config["lr"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLRonPlateau(
        optimizer, mode='min', factor=0.5, patience=2, threshold=1e-4
    )

    #replicate class imbalance handling via daaset inverse frequencies
    print("calculating class weights across the cascade network configurations")
    all_labels = []

    for _, labels in train_loader:
        all_labels.append(labels)
    class_counts = torch.bincount(torch.cat(all_labels))
    weights = 1.0 / (class_counts.float() + 1e-6)
    criterion = nn.CrossEntropyLoss(weight=(weights / weights.sum()).to(device))

    #metric containers
    history = {
        "train_loss": [], "train_acc": [], "train_prec": [], "train_rec": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_prec": [], "val_rec": [], "val_f1": [],
        "grad_norm": [], "purity": [], "epoch_time": [], "learning_rates": []
    }

    #fetch a fixed validation sample to trace state trajectory across epochs
    fixed_sample_batch = next(iter(val_loader))
    fixed_sample_img = fixed_sample_batch[0][0]

    for epoch in range(epochs):
        epoch_start = time.time()

        #epoch containers initialized natively
        train_conf_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int32, device=device)
        total_loss = 0.0
        epoch_purity = 0.0
        epoch_grad_norms = []


        #training phase
        model.train()
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [DV Train]")
        for batch_x, batch_y in train_loop:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            #physics audit 1: gradient norm tracker
            total_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            epoch_grad_norms.append(total_norm)

            #gradient clipping keeps parameter bounds
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            #accumulate values
            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)

            #vectorized confusion update running natively on GPU
            for t,p in zip(batch_y, preds):
                train_conf_matrix[t, p] += 1


            #physics audit 2: track state-vector purity 
            with torch.no_grad():
                latent = model.feature_extractor(batch_x)
                latent = torch.flatten(latent, 1)
                q_feats = model.quantum_projection_head(latent)

                state_dim = 2 ** model.n_qubits
                psi = torch.zeros(batch_x.size(0), state_dim, device=device, dtype=torch.complex64)

                psi[:, 0] = 1.0 + 0.0j
                evolved_psi = model.dv_quantum_classifier.quantum_ansatz.apply(psi, q_feats)

                for b in range(batch_x.size(0)):
                    purity = torch.real(torch.sum(torch.conj(evolved_psi[b]) * evolved_psi[b])).item()
                    epoch_purity += purity


        #validation phase
        model.eval()
        val_total_loss = 0.0

        val_conf_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int32, device=device)

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_total_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                for t, p in zip(batch_y, preds):
                    val_conf_matrix[t, p] += 1

        # state separation logics map
        save_qubit_trajectory(model, fixed_sample_img, epoch, run_dir, device)


        #process and compile statistical values
        total_train_samples = train_conf_matrix.sum().item()
        total_val_samples = val_conf_matrix.sum().item()

        avg_train_loss = total_loss / total_train_samples
        avg_val_loss = val_total_loss / total_val_samples
        avg_grad_norm = np.mean(epoch_grad_norms)

        #compute macro classifications via metrics
        train_acc, train_prec, train_rec, train_f1 = compute_metrics(*get_stats(train_conf_matrix))
        val_acc, val_prec, val_rec, val_f1 = compute_metrics(*get_stats(val_conf_matrix))

        epoch_duration = time.time() - epoch_start
        scheduler.step(avg_val_loss)
        current_lr = optimizer.params_groups[0]['lr']

        # Append to log history
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc.mean().item())
        history["train_prec"].append(train_prec.mean().item())
        history["train_rec"].append(train_rec.mean().item())
        history["train_f1"].append(train_f1.mean().item())
        
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc.mean().item())
        history["val_prec"].append(val_prec.mean().item())
        history["val_rec"].append(val_rec.mean().item())
        history["val_f1"].append(val_f1.mean().item())

        history["grad_norm"].append(avg_grad_norm)
        history["purity"].append(epoch_purity / total_train_samples)
        history["epoch_time"].append(epoch_duration)
        history["learning_rates"].append(current_lr)

        print(
            f"\nEpoch {epoch+1}/{epochs} Complete -> "
            f"Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1.mean().item():.3f} || "
            f"Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1.mean().item():.3f} || "
            f"Grad Norm: {avg_grad_norm:.3e} | Purity: {epoch_purity / total_train_samples:.5f} | "
            f"LR: {current_lr} | Time: {epoch_duration:.1f}s"
        )

    # Save model weights and parameter configurations to disk
    torch.save(model.state_dict(), os.path.join(run_dir, f"{model_name}.pt"))
    with open(os.path.join(run_dir, "metrics_history.json"), "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"\n[System] Weights and json metrics arrays successfully exported to: {run_dir}")
    return model, history
