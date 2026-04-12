import torch
import numpy as np
import os
import json
from experiments.metrics import compute_metrics
from experiments.plots import plot_inference_report_multiclass

def test(model, data, run_dir):
    model.eval()
    X_test, y_test = data['test']
    n_classes = data['n_classes']

    test_conf_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int32)
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for i in range(len(X_test)):
# 1. Forward pass (returns raw logits)
            # x_input = torch.tensor(X_test[i], dtype=torch.float32)
            logits, _ = model.forward(torch.tensor(X_test[i], dtype=torch.float32))
            # logits, _ = model.forward(torch.tensor(x_input, dtype=torch.float32))
            
            # 2. Get prediction index
            pred = torch.argmax(logits).item()
            target = int(y_test[i])
            
            test_conf_matrix[target, pred] += 1
            all_logits.append(logits.numpy())
            all_targets.append(int(y_test[i]))

    all_logits = np.array(all_logits)
    all_targets = np.array(all_targets)

    plot_inference_report_multiclass(all_targets, all_logits, run_dir, n_classes)

    # 3. Derive Multi-class TP, FP, TN, FN
    tp = torch.diag(test_conf_matrix).float()
    fp = test_conf_matrix.sum(dim=0).float() - tp
    fn = test_conf_matrix.sum(dim=1).float() - tp
    total_samples = test_conf_matrix.sum().float()
    tn = total_samples - (tp + fp + fn)

    # 4. Compute Metrics
    acc, prec, rec, f1 = compute_metrics(tp, fp, tn, fn)

    print(
        f"Test Final Metrics: || "
        f"Acc {acc.mean().item():.3f} | "
        f"Rec {rec.mean().item():.3f} | "
        f"Prec {prec.mean().item():.3f} | "
        f"F1 {f1.mean().item():.3f}"
    )

    # 6. Save JSON
    metrics = {
        "test_accuracy": acc.mean().item(),
        "test_precision": prec.mean().item(),
        "test_recall": rec.mean().item(),
        "test_f1": f1.mean().item(),
    }
    with open(os.path.join(run_dir, "test-metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics