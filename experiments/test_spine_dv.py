import torch
from tqdm import tqdm
from experiments.metrics import get_stats, compute_metrics

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