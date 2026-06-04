# Q57 — Binary Classical Continuous Benchmark

**Slice ID**: Q57-BINARY-CLASSICAL-CONTINUOUS-BENCHMARK  
**Campaign**: binary_classical_publication_package  
**Status**: BLOCKED  
**Depends on**: Q56 (BLOCKED on Q55)  
**Estimated runtime**: MEDIUM (20–60 min)  
**Date planned**: After Q56 completes  

---

## 1. Objective

Evaluate continuous/learned classifiers on the 128-dim Q49 MobileNetV3-Large embeddings.
Extends the Q56 discrete benchmark with gradient-based methods (gradient boosting, MLP heads)
and kernel methods (RBF SVM). Produces the Pareto frontier for AUROC vs parameter-count on the
continuous classifier family.

---

## 2. Inputs

Same as Q56: Q49 train/val/test embeddings at `workspace/experiments/Q49/embeddings/`.

---

## 3. Candidates

| Candidate ID | Classifier | Key Hyperparameters |
|---|---|---|
| svm_rbf_c1 | SVM (RBF kernel) | C=1.0, gamma=scale, probability=True |
| svm_rbf_c10 | SVM (RBF kernel) | C=10.0, gamma=scale, probability=True |
| xgb_default | XGBoost | n_estimators=200, max_depth=6, lr=0.1, seed=per_run |
| xgb_shallow | XGBoost | n_estimators=100, max_depth=3, lr=0.05, seed=per_run |
| lgbm_default | LightGBM | n_estimators=200, max_depth=-1, lr=0.05, seed=per_run |
| lgbm_shallow | LightGBM | n_estimators=100, num_leaves=31, lr=0.05, seed=per_run |
| mlp_1l_64 | MLP (1 hidden layer) | hidden=(64,), activation=relu, lr=1e-3, epochs=50 |
| mlp_1l_128 | MLP (1 hidden layer) | hidden=(128,), activation=relu, lr=1e-3, epochs=50 |
| mlp_2l_64_32 | MLP (2 hidden layers) | hidden=(64,32), activation=relu, lr=1e-3, epochs=50 |
| mlp_2l_128_64 | MLP (2 hidden layers) | hidden=(128,64), activation=relu, lr=1e-3, epochs=50 |

**MLP implementation notes:**
- Use `sklearn.neural_network.MLPClassifier` for consistency with Q56 environment
- Early stopping: `early_stopping=True, validation_fraction=0.1, n_iter_no_change=10`
- Applied to embeddings only — no backbone or feature extractor involved

---

## 4. Protocol

Identical to Q56 protocol:
- Seeds: [42, 7, 123]
- Fit on train; select on val; report final on test
- Metrics: AUROC, F1, Accuracy, Precision, Recall, latency_ms_per_sample

### 4.1 Pareto Frontier

After leaderboard is complete, compute the Pareto frontier on:
- x-axis: approximate parameter count (number of learnable parameters)
- y-axis: test AUROC (mean across seeds)

Pareto-dominant candidates are those with no other candidate simultaneously achieving higher AUROC
with fewer parameters. Report Pareto set in the summary report.

---

## 5. Reference Baselines

Same frozen baselines as Q56:

| Model | AUROC | F1 | Source |
|---|---|---|---|
| Q17 Classical CNN | 0.6224 | 0.5355 | Q17 |
| Q45A CLAHE baseline | 0.7196 | — | Q45A |
| Q46B MobileNetV3-Large (classical head trained) | 0.9873 | 0.9331 | Q46B/Q48 |
| Q56 winner (discrete classifier) | TBD | TBD | Q56 |

Q57 winner vs Q56 winner: establishes whether continuous/learned methods outperform discrete
classifiers on the frozen embedding space.

---

## 6. Outputs

| File | Description |
|---|---|
| `workspace/experiments/Q57/leaderboards/q57_continuous_benchmark_leaderboard.csv` | Per-seed + summary |
| `workspace/experiments/Q57/results/q57_continuous_benchmark_results.json` | Full results |
| `workspace/experiments/Q57/reports/q57_continuous_benchmark_report.md` | Analysis + Pareto set + winner |

---

## 7. Pass Criteria

- [ ] All 10 candidates × 3 seeds evaluated — 30 runs total
- [ ] Pareto frontier computed and reported (AUROC vs params)
- [ ] Q56 vs Q57 winner comparison included in report
- [ ] Leaderboard CSV with all required columns
- [ ] No source code modified; no git commit made
