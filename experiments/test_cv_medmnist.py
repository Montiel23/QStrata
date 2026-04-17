import torch
import numpy as np
import os
import json
from experiments.metrics import compute_metrics, calculate_purity
from experiments.plots import plot_inference_report_multiclass, plot_mode_wigner

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
    hbar = hbar
    purity = 0.0
    
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

            p = calculate_purity(cov, hbar)
            class_purity[target].append(p.item())

            #snr computation: mean squared/variance in mode 0
            snr = (mu[0]**2) / (cov[0,0] + 0.05)
            class_snr[target].append(snr.item())

            #store first correctly classified sample for wigner plotting
            if pred == target and target not in wigner_samples:
                wigner_samples[target] = (mu.clone(), cov.clone())

            test_conf_matrix[target, pred] += 1
            all_logits.append(logits.numpy())
            all_target.append(int(y_test[i]))


    # generate wigner plots
    wigner_dirs = os.path.join(run_dir, "wigners")
    os.makedirs(wigner_dir, exist_ok=True)
    for cls_id, (mu_w, cov_w) in wigner_sapmles.items():
        plot_mode_wigner(mu_w, cov_w, mode_idx=0,
                         save_path=os.path.join(wigner_dir, f"class_{cls_id}_wigner.png"))
        

    #aggregate metrics
    avg_purity = np.mean([np.mean(p) for p in class_purity.values() if p])
    avg_snr = {f"class_{c}_snr_dv": 10 * np.log10(np.mean(s)) for c, s in class_snr.items() if s}


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
        "test_accuracy": acc.mean().item(),
        "test_precision": prec.mean().item(),
        "test_recall": rec.mean().item(),
        "test_f1": f1.mean().item(),
        # "test_purity": purity.mean().item()
        "test_purity": avg_purity,
        **avg_snr
    }
    with open(os.path.join(run_dir, "test-metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics