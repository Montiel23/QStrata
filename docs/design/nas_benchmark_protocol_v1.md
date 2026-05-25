# NAS Benchmark Protocol v1

- **Status:** Active
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Supersedes:** Slice 18 benchmark approach (last-epoch, no seed)
- **Informed by:** Slice 19 audit findings

---

## Motivation

The Slice 18 benchmark sweep produced unstable results because metrics were recorded from the final training epoch only and no random seed was fixed, causing results to vary significantly between runs depending on whether the optimizer happened to land in a stable or spike-affected state at the last epoch. The Slice 19 audit confirmed that training accuracy peaked mid-run and decayed by the final epoch — in one case C001 reached val_acc 91.79% at epoch 7 before collapsing to 77.67% at epoch 10, and Slice 18 reported an even more extreme collapse to 33.97%. This protocol defines the standard all future NAS v0 benchmarks must follow to produce reproducible and trustworthy results.

---

## Protocol Requirements

1. **Fixed random seed** — seed 42 applied to Python `random`, NumPy, and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`) before any data loading or model instantiation; `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` must be set at the same time.

2. **Epoch-level validation tracking** — validation accuracy must be evaluated and recorded after every training epoch; a per-epoch log showing train loss, val loss, train accuracy, val accuracy, and epoch time must be printed.

3. **Best-checkpoint selection** — model weights must be saved in memory (no file I/O required) at the epoch that achieves the highest validation accuracy observed so far; the best epoch index and its validation accuracy must be tracked explicitly.

4. **Test evaluation at best checkpoint** — after training completes, the model weights must be restored to the best-validation checkpoint before any test set evaluation; test accuracy must not be evaluated using final-epoch weights.

5. **Reported metrics** — the following metrics must be collected and reported for each benchmarked config: parameter count, best validation accuracy, best epoch, final train accuracy, test accuracy at best checkpoint, mean epoch time, and mean inference latency.

---

## Metric Definitions

- **Best validation accuracy** — highest validation accuracy recorded across all training epochs.
- **Best epoch** — the epoch index (1-based) at which best validation accuracy occurred.
- **Final train accuracy** — training accuracy recorded at the last epoch.
- **Test accuracy** — accuracy on the held-out test split, evaluated using the weights from the best-validation epoch; test accuracy must not be used as a NAS fitness signal.
- **Mean epoch time** — wall-clock mean across all training epochs, in seconds.
- **Mean inference latency** — total wall-clock time for a full test set forward pass in `eval()` mode using best-validation weights, divided by the number of batches, in ms/batch.

---

## Fitness Signal Rule

Validation accuracy — specifically, best validation accuracy across all training epochs — is the sole fitness signal for NAS search. Test accuracy is reported for analysis and final evaluation only and must never be used to select or rank candidates during search.

---

## Applies To

This protocol applies to:
- All future NAS v0 candidate benchmarks
- Any re-run of Slice 18 candidates using the stable script (`benchmark_nas_v0_candidates_stable.py`)
- Any benchmark that feeds results into NAS search space decisions
