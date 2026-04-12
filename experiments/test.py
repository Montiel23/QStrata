import torch
import numpy as np
import os
import json
from experiments.metrics import compute_metrics
from experiments.plots import plot_boundary

def test(model, X_test, y_test, run_dir):
    model.eval()
    all_probs = []
    all_preds = []

    tp = fp = tn = fn = 0

    with torch.no_grad():
        for i in range(len(X_test)):
            p, _ = model.forward(X_test[i])
            prob = p.item()
            all_probs.append(prob)
            all_preds.append(int(1.0 if prob > 0.5 else 0.0))

            pred = (p > 0.5).float()
            if pred == 1 and y_test[i] == 1:
                tp += 1
            elif pred == 1 and y_test[i] == 0:
                fp += 1
            elif pred == 0 and y_test[i] == 0:
                tn += 1
            elif pred == 0 and y_test[i] == 1:
                fn += 1

    acc, prec, rec, f1 = compute_metrics(tp, fp, tn, fn)
    y_true = y_test.numpy()
    y_probs = np.array(all_probs)

    print(
        f"Test metrics: ||"
        f"Acc {acc:.3f} |"
        f"Rec {rec:.3f} |"
        f"Prec {prec:.3f} |"
        f"F1 {f1:.3f} |"
    )

    plot_boundary(model, X_test, y_test, "test-decision_boundary", run_dir)
    
    # save metrics
    metrics = {
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
    }
        
    with open(os.path.join(run_dir, "test-metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return y_true, y_probs