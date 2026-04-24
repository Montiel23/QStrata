import torch
import numpy as np
import os
import json
from experiments.metrics import compute_metrics, calculate_purity, compute_gaussian_fidelity, analyze_state_separation
from experiments.plots import plot_inference_report_multiclass, plot_mode_wigner, plot_fidelity_matrix

def test(model, data, run_dir, hbar=2.0):
    model.eval()
    X_test, y_test = data["test"]
    n_classes = data["n_classes"]

    test_conf_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int32)
    class_purity = {c: [] for c in range(n_classes)}
    class_snr = {c: [] for c in range(n_classes)}


    wigner_samples = {}
    
    all_logits = []
    all_targets = []
    purity = 0.0

    test_results = {"mus": [], "covs": [], "labels": []}


    
    with torch.no_grad():
        for i in range(len(X_test)):
            x_input = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0)
            logits = model(x_input)
            # logits = model(X_test[i], dtype=torch.float32)

            pred = torch.argmax(logits).item()
            target = int(y_test[i])


            #physics audit
            mu, cov  = model.backend.get_vacuum()
            mu, cov = model.ansatz.apply(mu, cov, model.backend)

            test_results["mus"].append(mu.detach().cpu().numpy())
            test_results["covs"].append(cov.detach().cpu().numpy())
            test_results["labels"].append(target)
            

            p = calculate_purity(cov, hbar)
            class_purity[target].append(p.item())

            #snr computation: mean squared/variance in mode 0
            snr = (mu[0]**2) / (cov[0,0] + 0.05)
            class_snr[target].append(snr.item())

            #store first correctly classified sample for wigner plotting
            if pred == target and target not in wigner_samples:
                wigner_samples[target] = (mu.clone(), cov.clone())

            test_conf_matrix[target, pred] += 1
            # all_logits.append(logits.numpy())
            all_logits.append(logits.detach().cpu().numpy().squeeze())
            all_targets.append(int(y_test[i]))


    # generate wigner plots
    wigner_dirs = os.path.join(run_dir, "wigners")
    os.makedirs(wigner_dirs, exist_ok=True)
    for cls_id, (mu_w, cov_w) in wigner_samples.items():
        plot_mode_wigner(mu_w, cov_w, mode_idx=0,
                         save_path=os.path.join(wigner_dirs, f"class_{cls_id}_wigner.png"))
        

    #aggregate metrics
    avg_purity = np.mean([np.mean(p) for p in class_purity.values() if p])
    avg_snr = {f"class_{c}_snr_cv": 10 * np.log10(np.mean(s) + 1e-9) for c, s in class_snr.items() if s}


    all_logits = np.array(all_logits)
    all_targets = np.array(all_targets)

    plot_inference_report_multiclass(all_targets, all_logits, run_dir, n_classes)

    tp = torch.diag(test_conf_matrix).float()
    fp = test_conf_matrix.sum(dim=0).float() - tp
    fn = test_conf_matrix.sum(dim=1).float() - tp
    total_samples = test_conf_matrix.sum().float()
    tn = total_samples - (tp + fp + fn)

    # 4. Compute Metrics
    acc, prec, rec, f1 = compute_metrics(tp, fp, tn, fn)
    # _, last_conv = model.backend.get_vacuum()
    # purity = calculate_purity(last_conv, hbar)

    print(
        f"Test Final Metrics: || "
        f"Acc {acc.mean().item():.3f} | "
        f"Rec {rec.mean().item():.3f} | "
        f"Prec {prec.mean().item():.3f} | "
        f"F1 {f1.mean().item():.3f} |"
        f"Purity {avg_purity:.3f}"
        # f"Purity {purity.mean().item():.3f} |"
    )

    # 6. Save JSON
    metrics = {
        "test_accuracy": float(acc.mean().item()),
        "test_precision": float(prec.mean().item()),
        "test_recall": float(rec.mean().item()),
        "test_f1": float(f1.mean().item()),
        # "test_purity": purity.mean().item()
        "test_purity": float(avg_purity),
        **{k: float(v) for k, v in avg_snr.items()}
    }
    with open(os.path.join(run_dir, "test-metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, test_results


def run_minimal_val(model, data, config, run_dir, noise_range, squeezing_range):
    """
    Computes a full Phase Diagram (F1-score grid) across noise and squeezing levels.
    """
    X_test, y_test = data["test"]

    X_test_input = torch.tensor(X_test, dtype=torch.float32)
    y_test_input = torch.tensor(y_test, dtype=torch.long)

    results = []
    # Save original weights to restore them after the stress test
    original_weights = {n: p.clone() for n, p in model.named_parameters()}

    # 1. Start the nested physics sweep
    for noise in noise_range:
        for s_limit in squeezing_range:
            # Inject current hardware constraints
            model.hbar = noise
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "squeezing" in name:
                        param.clamp_(-s_limit, s_limit)

            # 2. Evaluate performance under these constraints
            tp = fp = tn = fn = 0
            model.eval()
            with torch.no_grad():
                # for i in range(len(X_test)):
                for i in range(len(X_test_input)):
                    # Note: Using unsqueeze(0) for batch consistency
                    # logits = model(X_test[i].unsqueeze(0)) 
                    logits = model(X_test_input[i].unsqueeze(0)) 
                    pred = torch.argmax(logits, dim=1).item()
                    # target = y_test[i].item()
                    target = y_test_input[i].item()

                    # Simple binary metric logic (adapt for multiclass if needed)
                    if pred == 1 and target == 1: tp += 1
                    elif pred == 1 and target == 0: fp += 1
                    elif pred == 0 and target == 0: tn += 1
                    elif pred == 0 and target == 1: fn += 1

            # 3. Store result for this specific coordinate in phase space
            acc, prec, rec, f1 = compute_metrics(tp, fp, tn, fn)
            results.append({'noise': noise, 's_limit': s_limit, 'f1': f1.item() if torch.is_tensor(f1) else f1})

            # 4. RESET: Crucial to restore weights for the next (noise, s_limit) pair
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(original_weights[name])

    # 5. Return the full list of results instead of just one number
    return results

# def run_minimal_val(model, X_test, y_test, config, run_dir, noise_range, squeezing_range):
#     all_probs = []
#     all_preds = []

#     tp = fp = tn = fn = 0

#     model.hbar = config.noise


#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if "squeezing" in name:
#                 param.clamp_(-config.max_squeezing, config.max_squeezing)

#     model.eval()

#     with torch.no_grad():
#         for i in range(len(X_test)):
#             p, _ = model.forward(X_test[i])
#             prob = p.item()
#             all_probs.append(prob)
#             all_preds.append(int(1.0 if prob > 0.5 else 0.0))

#             pred = (p > 0.5).float()
#             if pred == 1 and y_test[i] == 1:
#                 tp += 1
#             elif pred == 1 and y_test[i] == 0:
#                 fp += 1

#             elif pred == 0 and y_test[i] == 0:
#                 tn += 1
            
#             elif pred == 0 and y_test[i] == 1:
#                 fn += 1

#     acc, prec, rec, f1 = compute_metrics(tp, fp, tn, fn)

#     y_true = y_test.numpy()
#     y_probs = np.array(all_probs)

#     return f1