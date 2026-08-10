import torch
from tqdm import tqdm
from experiments.metrics import get_stats, compute_metrics
import numpy as np

def test_spine_dv_engine(model, data_loader, criterion, n_classes, device):
    model.eval()
    test_loss = 0.0

    all_y_true = []
    all_y_logits = []

    conf_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int32, device=device)

    progress_bar = tqdm(data_loader, desc="Evaluating test partition", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            if torch.norm(images) == 0:
                continue
            
            inputs = images.to(device)
            targets = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            test_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)

            #vectorized confusion matrix update on the GPU
            for t, p in zip(targets, preds):
                conf_matrix[t, p] += 1

            all_y_true.extend(labels.numpy())
            all_y_logits.extend(logits.cpu().numpy())

    total_samples = conf_matrix.sum().item()
    avg_loss = test_loss / total_samples if total_samples > 0 else 0.0

    tp, fp, tn, fn = get_stats(conf_matrix)
    acc, prec, rec, f1 = compute_metrics(tp, fp, tn, fn)

    metrics_summary = {
        "loss": avg_loss,
        "accuracy": acc.mean().item(),
        "precision": prec.mean().item(),
        "recall": rec.mean().item(),
        "f1_score": f1.mean().item(),
        "confusion_matrix": conf_matrix.cpu().numpy()
    }

    return metrics_summary, all_y_true, all_y_logits


# def test_spine_cv(model, test_loader, n_classes, class_names, run_dir, device, hbar=1.0):
#     print("\n------------------")
#     print("evaluating unseen test generalization")
#     print("---------------------")

#     model.eval()
#     all_logits = []
#     all_targets = []
#     test_conf_matrix = torch.zeros(
#         (n_classes, n_classes), dtype=torch.int32, device=device
#     )

#     with torch.no_grad():
#         for batch_x, batch_y in test_loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             logits = model(batch_x)

#             preds = torch.argmax(logits, dim=1)
#             for t, p in zip(batch_y, preds):
#                 test_conf_matrix[t, p] += 1

#             all_logits.append(logits.cpu().numpy())
#             all_targets.append(batch_y.cpu().numpy())

#     test_logits = np.concatenate(all_logits, axis=0)
#     test_targets = np.concatenate(all_targets, axis=0)

#     test_acc, test_prec, test_rec, test_f1 = compute_metrics(
#         *get_stats(test_conf_matrix)
#     )

#     print("\n" + "=" * 50)
#     print("               TEST EVALUATION METRICS")
#     print("=" * 50)
#     print(f"Accuracy  : {test_acc.mean().item():.4f}")
#     print(f"Precision : {test_prec.mean().item():.4f}")
#     print(f"Recall    : {test_rec.mean().item():.4f}")
#     print(f"F1-Score  : {test_f1.mean().item():.4f}")
#     print("=" * 50 + "\n") 