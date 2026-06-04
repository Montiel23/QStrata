# Q56 — Binary Classical Discrete Benchmark

**Slice ID**: Q56-BINARY-CLASSICAL-DISCRETE-BENCHMARK  
**Campaign**: binary_classical_publication_package  
**Status**: BLOCKED  
**Depends on**: Q55 (READY)  
**Estimated runtime**: MEDIUM (15–45 min)  
**Date planned**: After Q55 completes  

---

## 1. Objective

Evaluate a suite of discrete (traditional/non-gradient) classical ML classifiers on the 128-dim
MobileNetV3-Large embeddings from Q49. Establishes the classical ML ceiling without neural network
training — quantifies how much discriminative power is already in the frozen embedding space.

---

## 2. Inputs

| Artifact | Path | Notes |
|---|---|---|
| Train embeddings | `workspace/experiments/Q49/embeddings/train_embeddings.npy` | (6712, 128) |
| Train labels | `workspace/experiments/Q49/embeddings/train_labels.npy` | (6712,) |
| Val embeddings | `workspace/experiments/Q49/embeddings/val_embeddings.npy` | (1677, 128) |
| Val labels | `workspace/experiments/Q49/embeddings/val_labels.npy` | (1677,) |
| Test embeddings | `workspace/experiments/Q49/embeddings/test_embeddings.npy` | (2077, 128) |
| Test labels | `workspace/experiments/Q49/embeddings/test_labels.npy` | (2077,) |

---

## 3. Candidates

| Candidate ID | Classifier | Key Hyperparameters |
|---|---|---|
| lr_l2 | Logistic Regression (L2) | C=1.0, max_iter=1000, solver=lbfgs |
| lr_l1 | Logistic Regression (L1) | C=1.0, max_iter=1000, solver=saga |
| svm_linear | SVM (linear kernel) | C=1.0, probability=True |
| knn_5 | k-Nearest Neighbors | k=5, metric=euclidean |
| knn_15 | k-Nearest Neighbors | k=15, metric=euclidean |
| nb_gaussian | Naive Bayes (Gaussian) | var_smoothing=1e-9 |
| dt_gini | Decision Tree | criterion=gini, max_depth=10 |
| rf_100 | Random Forest | n_estimators=100, max_depth=None |
| rf_200 | Random Forest | n_estimators=200, max_depth=15 |

---

## 4. Protocol

### 4.1 Seeds and Evaluation

- Seeds: [42, 7, 123] — applied to random-state parameters where applicable
- Fit on train embeddings only; evaluate on val (model selection) and test (final reporting)
- Test metrics are recorded but not used for model selection (consistent with QStrata protocol)

### 4.2 Metrics (per split: val and test)

| Metric | Notes |
|---|---|
| AUROC | Primary — area under ROC curve; `sklearn.metrics.roc_auc_score` |
| F1 | Macro-averaged binary F1; `sklearn.metrics.f1_score(average='binary')` |
| Accuracy | `sklearn.metrics.accuracy_score` |
| Precision | `sklearn.metrics.precision_score` |
| Recall | `sklearn.metrics.recall_score` |
| Inference latency (ms/sample) | Wall time for test-set prediction / N_test |

### 4.3 Leaderboard Format

CSV columns: `rank, candidate_id, seed, val_auroc, val_f1, val_accuracy, test_auroc, test_f1, test_accuracy, test_precision, test_recall, latency_ms_per_sample`

Summary rows (mean ± std across seeds) added after per-seed rows.

### 4.4 Decision Rule

Winner = highest mean val_auroc across seeds. Ties broken by mean val_f1.

---

## 5. Reference Baselines

The following frozen benchmarks are the comparison floor:

| Model | AUROC | F1 | Source |
|---|---|---|---|
| Q17 Classical CNN (full network training) | 0.6224 | 0.5355 | Q17 |
| Q45A CLAHE baseline (no augmentation) | 0.7196 | — | Q45A |
| Q46B MobileNetV3-Large (frozen, classical head) | 0.9873 | 0.9331 | Q46B/Q48 |

Q56 classifiers operate on the Q49 embeddings (MobileNetV3-Large features). Comparing to Q46B
establishes how much the full head training contributes vs. a linear probe on frozen features.

---

## 6. Outputs

| File | Description |
|---|---|
| `workspace/experiments/Q56/leaderboards/q56_discrete_benchmark_leaderboard.csv` | Per-seed + summary leaderboard |
| `workspace/experiments/Q56/results/q56_discrete_benchmark_results.json` | Full results dict |
| `workspace/experiments/Q56/reports/q56_discrete_benchmark_report.md` | Analysis and winner declaration |

---

## 7. Pass Criteria

- [ ] All 9 candidates × 3 seeds evaluated — 27 runs total
- [ ] Leaderboard CSV written with all required columns
- [ ] Winner declared with val_auroc > Q45A baseline (0.7196) to confirm embeddings are useful
- [ ] Reference baselines tabulated in report
- [ ] No source code modified; no git commit made
