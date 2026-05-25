# NAS v0 Benchmark Audit Report

- **Status:** Complete
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Trigger:** Anomalous C001 results in Slice 18

---

## Section 1: Observed Anomaly Summary

C001 (`standard [32,64]`, `binary_baseline.yaml`) has prior validated test accuracy of 85.74%–88.62% and validation accuracy in the 89–92% range across Slices 12 and 13. Slice 18 reported a final val accuracy of 33.97% and test accuracy of 40.87% for C001 — a drop of more than 50 percentage points from the prior validated baseline. The Slice 18 per-epoch log reveals that C001 reached val_acc 0.9027 at epoch 9 but then suffered a catastrophic val_loss spike (0.3633 → 1.6701) at epoch 10, leaving the model weights in a severely degraded state that the test evaluation then measured. C006 (`depthwise_sep [64,128]`) showed the same pattern — peaking at val_acc 0.8969 at epoch 9 before collapsing to 0.4580 at epoch 10. C005 also ended below its mid-training peak, suggesting the instability is not confined to a single candidate. The pattern is consistent with Adam optimizer instability on a small dataset in the absence of a learning-rate scheduler or gradient clipping: the final-epoch state is not reliably representative of the model's trained capability.

---

## Section 2: Baseline Re-Run Comparison

| Metric | Prior validated baseline | Slice 18 result | Audit re-run |
|---|---:|---:|---:|
| Params | 19,138 | 19,138 | 19,138 |
| Final train accuracy | ~90% | 89.97% | 89.89% |
| Final val accuracy | ~89–90% | 33.97% | 77.67% |
| Test accuracy | ~85–88% | 40.87% | 77.24% |
| Mean latency (ms/batch) | ~0.61–0.71 | 0.59 | 0.65 |

The audit re-run confirms the instability hypothesis: the model again failed to reach the prior validated accuracy range at the final epoch (val_acc 77.67%, test_acc 77.24%), even though the per-epoch log shows it peaked at val_acc 0.9179 at epoch 7 before losing stability at epochs 8–10 — demonstrating that the pipeline is structurally sound but that last-epoch metric reporting is unreliable without a fixed random seed.

---

## Section 3: Audit Findings Checklist

- [PASS] Same dataset split used across all candidates (train/val/test not shuffled or swapped)?
  Evidence: `get_dataset("pneumoniamnist", "train")`, `get_dataset("pneumoniamnist", "val")`, and `get_dataset("pneumoniamnist", "test")` are called as three separate, explicitly named invocations in `run_candidate()`. The medmnist library resolves these to fixed NPZ array slices. The audit re-run confirmed split sizes: train=4708, val=524, test=624 — consistent with all prior runs.

- [SUSPICIOUS] Random seed behavior consistent and not interfering with split reproducibility?
  Evidence: No random seed is set anywhere in the pipeline — not in the script, not in the data loading, not in model initialisation. Split *assignment* is deterministic (medmnist NPZ boundaries are fixed), but mini-batch *ordering* varies per run due to `DataLoader(shuffle=True)` and model weight initialisation varies per run via PyTorch's default seeding. This does not cause split contamination but makes training dynamics non-reproducible across runs; any epoch can be a spike epoch, and the final epoch may or may not land on a stable state.

- [PASS] Class weights applied correctly per the YAML config?
  Evidence: `compute_class_weights(train_adapted, num_classes, device)` is called when `use_cw=True`, computed from the training split only. The weights [1.939, 0.674] are mathematically consistent with the class distribution (class0=1214, class1=3494 in train) and match across all runs.

- [PASS] Loss function matches the baseline config?
  Evidence: `nn.CrossEntropyLoss(weight=cw)` is used. This matches the binary classification setup specified in the config. No alternative loss function is present.

- [PASS] Optimizer config matches the baseline config?
  Evidence: `torch.optim.Adam(model.parameters(), lr=lr)` where `lr = config["training"]["lr"] = 0.001`. No momentum or weight-decay overrides. Matches the baseline config exactly.

- [PASS] Epoch count matches the baseline config?
  Evidence: `epochs = config["training"]["epochs"]` reads directly from the YAML. For all six candidates the value is 10. Confirmed from epoch logs in Slice 18 and the audit re-run.

- [PASS] Model built using the correct `build_model()` call with correct `block_type` and `conv_channels`?
  Evidence: `build_model(model_cfg)` is called with a dict assembled from `config["model"]` and `config["dataset"]`. For C001 (`binary_baseline.yaml`), the model section has no `block_type` key; `build_model()` defaults to `"standard"` via `config.get("block_type", "standard")`. The audit re-run printed the architecture and confirmed `Conv2d(1, 32) → BN → ReLU → Conv2d(32, 64) → BN → ReLU → AdaptiveAvgPool2d → Flatten → Dropout(0.3) → Linear(64, 2)` with 19,138 parameters — exactly as expected.

- [PASS] Config parsed correctly — no field silently dropped or defaulted incorrectly?
  Evidence: `yaml.safe_load()` is used. The audit re-run printed the parsed config dict and confirmed all fields: `conv_channels=[32,64]`, `dropout=0.3`, `use_batchnorm=True`, `pooling=adaptive_avg`, `epochs=10`, `batch_size=64`, `lr=0.001`, `class_weights=True`, `input_channels=1`, `num_classes=2`. No silent defaults or missing fields observed.

- [PASS] Metric logging path correct — train/val/test metrics not cross-assigned?
  Evidence: `final_train_acc` is updated inside the training loop from `n_cor / n_tot` (training batches only). `final_val_acc` is updated inside the validation loop from `v_cor / v_tot` (validation batches only). `test_acc` is computed from `t_cor / t_tot` in a separate post-training test loop. No cross-assignment path exists in the code.

- [PASS] Validation metric computed on the correct split?
  Evidence: The validation loop iterates over `val_loader`, which wraps `val_adapted`, which wraps `get_dataset(..., "val")`. The audit re-run confirmed the val split contains 524 samples (135 class0, 389 class1), consistent with the medmnist PneumoniaMNIST val split specification.

- [PASS] Test metric computed on the correct split?
  Evidence: The test loop iterates over `test_loader`, which wraps `test_adapted`, which wraps `get_dataset(..., "test")`. The audit re-run confirmed the test split contains 624 samples (234 class0, 390 class1), consistent with the medmnist PneumoniaMNIST test split specification.

- [PASS] No label leakage between splits?
  Evidence: Each of the three splits is loaded by a separate `get_dataset()` call with a distinct `split` string. Class weights are computed from `train_adapted` only. The val and test loaders are never passed to the training loop. No data augmentation or mixing across splits.

- [PASS] Correct split loaded for each phase (train loader used only for training, val loader used only for validation)?
  Evidence: In `run_candidate()`, `train_loader` appears only in the training `for xb, yb in train_loader` loop (which calls `optimizer.step()`). `val_loader` appears only in the validation loop (which is under `torch.no_grad()`). `test_loader` appears only in the post-training test evaluation block. No loader is used in the wrong phase.

- [PASS] Model in `eval()` mode during validation and test evaluation?
  Evidence: `model.eval()` is called at the top of the per-epoch validation block (before `torch.no_grad()`). For test evaluation, `model.eval()` is called once before the test loop. Confirmed in both `benchmark_nas_v0_candidates.py` and `audit_nas_v0_benchmark.py`.

- [PASS] BatchNorm running stats in correct state during evaluation?
  Evidence: `model.eval()` correctly switches BatchNorm layers to use accumulated running mean and variance rather than batch statistics. No manual BN state manipulation is present. This is standard PyTorch behaviour and is functioning correctly.

- [PASS] No accidental model weight overwrite or reset between candidates?
  Evidence: Each call to `run_candidate()` begins with `model = build_model(model_cfg).to(device)` — a fresh instantiation. No global model variable is shared between candidates. The `optimizer = torch.optim.Adam(model.parameters(), lr=lr)` is also re-created per candidate. No state leaks between runs.

---

## Section 4: Conclusion

1. **Are the Slice 18 benchmark results trustworthy?** No — the results for C001 and C006 are not trustworthy as representations of trained model capability, because both candidates' final-epoch states were dominated by training instability spikes rather than converged performance; C005 is also suspect for the same reason, and the remaining candidates may have happened to land on stable final epochs by chance rather than by reproducible convergence.

2. **Is there a reproducibility bug in the benchmark pipeline?** No structural bug was found in data loading, split assignment, model construction, or metric attribution; the issue is that the measurement methodology — recording last-epoch validation accuracy without a fixed random seed, a learning-rate scheduler, or best-epoch tracking — is unreliable on this dataset and produces results that vary significantly between runs depending on whether the final epoch falls on a stable or spike-affected state.

3. **Can NAS proceed safely on the basis of current results?** No — before NAS proceeds, the benchmark methodology must be made reliable: either a fixed random seed must be established for reproducibility, or best-epoch validation accuracy must be recorded rather than last-epoch accuracy, or both; the current results cannot serve as a trustworthy fitness signal for multi-objective search because the same candidate can report anywhere from ~34% to ~93% validation accuracy across runs depending on final-epoch luck.
