# QStrata Experiment Configuration Schema

**Project:** QStrata — medical imaging model optimization R&D  
**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-26  
**Author:** Miguel Lopez (QStrata)  
**Slice:** Q30 — Experiment Automation Framework Design; Q31A — formalized command block  
**Status:** IMPLEMENTED (Q31); command block formalized (Q31A)

---

## 1. Purpose

Standardized experiment configurations serve three functions simultaneously:

1. **Reproducibility.** A frozen config + git commit SHA is sufficient to reconstruct any experiment. No external state, undocumented defaults, or environment-variable overrides should be needed.

2. **NAS compatibility.** NAS trial generation (Q32, Q33, Q34) produces experiment configs programmatically. The schema must be rich enough to describe any trial and simple enough that a search algorithm can generate valid configs without access to implementation details.

3. **Immutable experiment tracking.** Once an experiment starts, its config is frozen and linked to its results by `experiment_id`. All reports and leaderboards cite `experiment_id` rather than free-form descriptions. The config is the authoritative definition of what ran.

This document defines the YAML schema for QStrata experiment configs. All experiments run through the Q31 runner must conform to this schema. Configs that fail schema validation are rejected before execution.

---

## 2. Schema Overview

The config is divided into seven top-level sections: `experiment`, `dataset`, `model`, `training`, `metrics`, `artifacts`, `reproducibility`, and `command`. The `command` block was added in Q31 as a required field and formalized in Q31A. Each section is described below with field definitions, required status, and mutability rules.

---

## 3. Full YAML Schema

```yaml
experiment:
  experiment_id:       # auto-generated at run start: <ISO_timestamp>_<8char_hash>
                       # e.g. 20260526T143021_a3f7c2b1
                       # immutable once set; never manually specified
  phase:               # string: binary_dv | binary_cv | classical_nas | quantum_nas_dv |
                       #         quantum_nas_cv | multiclass | ablation | custom
                       # immutable once experiment starts
  description:         # optional free-text human label; not used in analysis
                       # mutable; does not affect results

dataset:
  name:                # string: dataset identifier, e.g. vindr_binary_roi_224
  version:             # string: dataset version tag, e.g. v1
  split:               # string or dict: e.g. "canonical_42" or
                       #   {train: 0.70, val: 0.15, test: 0.15, seed: 42}
  preprocessing:       # string: reference to preprocessing config hash or profile name
                       # e.g. "roi_224_normalize_grayscale"

model:
  architecture:        # string: architecture family identifier
                       # e.g. dv_hybrid | cv_hybrid | classical_control | classical_nas_trial
  backbone:            # string: backbone checkpoint identifier or null for random init
                       # e.g. c006_d040_classical_anchor | null
  backbone_frozen:     # bool: whether backbone weights are frozen during training
  head:                # string: head type identifier
                       # e.g. DVQuantumHead | CVGaussianHead | TinyClassicalHead | NASTrialHead
  parameters:          # dict: head and architecture hyperparameters
                       # varies by architecture; NAS trials vary this block
                       # all keys affecting results must appear here
                       # example for DV hybrid:
                       #   n_qubits: 4
                       #   circuit_depth: 1
                       #   ansatz_type: rotational
                       #   compression_dim: 4
                       # example for CV hybrid:
                       #   n_modes: 2
                       #   cv_depth: 1
                       #   squeezing_cap: 1.5
                       #   hbar: 2.0
                       #   compression_dim: 4
                       # example for classical NAS trial:
                       #   block_type: depthwise_sep
                       #   channels: [64, 128]
                       #   depth: 2
                       #   pooling: adaptive_avg

training:
  epochs:              # int: maximum training epochs
  batch_size:          # int: samples per gradient step
  optimizer:           # string: AdamW | Adam | SGD
  learning_rate:       # float: initial learning rate
  weight_decay:        # float: L2 regularization coefficient
  loss:                # string: CrossEntropyLoss | WeightedCrossEntropyLoss
  class_weights:       # list or null: per-class loss weights; null = unweighted
  early_stopping:
    enabled:           # bool
    monitor:           # string: val_loss | val_auroc (minimize or maximize respectively)
    patience:          # int: epochs without improvement before stopping
    mode:              # string: min | max (direction of improvement)
  lr_schedule:         # string or null: CosineAnnealing | StepLR | null
  augmentation:        # string or null: reference to augmentation profile or null

metrics:
  primary:             # string: metric name used for checkpoint selection and leaderboard ranking
                       # e.g. val_auroc
  secondary:           # list of strings: additional tracked metrics
                       # e.g. [val_f1, val_acc, val_auprc, val_precision, val_recall]
  cv_health:           # bool: whether to run per-epoch CV health checks (COV_PSD etc.)
                       # only applicable to CV hybrid architectures

artifacts:
  save_best_checkpoint:   # bool: save checkpoint at best val metric
  save_final_checkpoint:  # bool: save checkpoint at last epoch
  save_confusion_matrix:  # bool: export confusion matrix at test eval
  save_roc_curve:         # bool: export ROC curve at test eval
  export_per_epoch_metrics: # bool: write per-epoch CSV to results dir

reproducibility:
  seed:                # int: random seed; set for Python random, numpy, torch before any op
  git_commit:          # string: auto-populated from git rev-parse HEAD at run start
                       #         value 'dirty' if working tree has uncommitted changes
  hardware:            # dict: auto-populated at run start
                       #   gpu_model: string
                       #   cuda_version: string
                       #   cpu_fallback: bool
                       #   gpu_memory_mb: int

command:
  executable:          # REQUIRED string: subprocess executable name or path
                       # e.g. "python3", "bash"
                       # The runner invokes [executable] + args as a subprocess
                       # Use "python3" (not "python") for containers where only python3 is in PATH
  args:                # REQUIRED list of strings: script path and arguments
                       # e.g.
                       #   - scripts/smoke_test_vindr_cv_binary.py
                       #   - --root
                       #   - data/processed/vindr_binary_roi_224
                       #   - --seed
                       #   - "42"
                       # All args must be strings; numeric values must be quoted
```

---

## 4. Required Fields and Mutability

| Field | Required | Immutable Once Started | Notes |
|---|---|---|---|
| `experiment.experiment_id` | Yes (auto) | Yes | Never manually set; auto-generated at run start |
| `experiment.phase` | Yes | Yes | Determines leaderboard assignment |
| `experiment.description` | No | No | Human label only; not used in comparisons |
| `dataset.name` | Yes | Yes | Changes to dataset require new experiment |
| `dataset.version` | Yes | Yes | Version pinned to dataset state |
| `dataset.split` | Yes | Yes | Split definition is part of experiment identity |
| `dataset.preprocessing` | Yes | Yes | Preprocessing changes require new experiment |
| `model.architecture` | Yes | Yes | Architecture family is immutable |
| `model.backbone` | Yes | Yes | Backbone identity is immutable |
| `model.backbone_frozen` | Yes | Yes | Freeze state affects all gradient computations |
| `model.head` | Yes | Yes | Head type is immutable |
| `model.parameters` | No | Yes if provided | Any subset of parameters provided is frozen |
| `training.seed` | Yes | Yes | Seed is part of experiment identity |
| `training.epochs` | Yes | No | May be extended for resumed runs |
| `training.batch_size` | Yes | Yes | Changes training dynamics |
| `training.learning_rate` | Yes | Yes | Core hyperparameter |
| `training.optimizer` | Yes | Yes | Optimizer choice is frozen |
| `training.early_stopping` | Yes | Yes | Stopping criterion is frozen |
| `metrics.primary` | Yes | Yes | Cannot change selection criterion mid-run |
| `metrics.secondary` | Yes | No | Additional metrics may be added to resumed runs |
| `reproducibility.seed` | Yes | Yes | Identical to `training.seed`; stored redundantly for clarity |
| `reproducibility.git_commit` | Yes (auto) | Yes | Populated at run start; immutable |
| `reproducibility.hardware` | No (auto) | Yes if present | Populated at run start |

---

## 5. Artifact Tracking Rules

The following artifact tracking rules are enforced by the runner:

**Metric export (partial, per-epoch):**
- At the end of each training epoch, all metrics computed so far are written atomically to `experiments/results/<experiment_id>_partial.json`
- Write is atomic: written to `<experiment_id>_partial.json.tmp` then renamed
- This ensures a hardware failure mid-epoch does not corrupt the previous epoch's record

**Metric export (final):**
- On experiment completion (`completed` or `failed`), all metrics are written to `experiments/results/<experiment_id>.json`
- This file is set to read-only after writing: `chmod 444`
- The partial file is deleted after the final file is written

**Config freezing:**
- Before any other operation, the config YAML is validated against this schema
- Validated config is written to `experiments/configs/<experiment_id>.yaml`
- File permissions set to read-only: `chmod 444`
- SHA-256 hash of the frozen config is recorded in the experiment metadata
- The runner raises an error if the frozen config file already exists (prevents double-start)

**Checkpoint naming:**
- Best checkpoint: `experiments/checkpoints/<experiment_id>/best.pt`
- Final checkpoint: `experiments/checkpoints/<experiment_id>/final.pt`
- Checkpoint directory is created at experiment start; cleared on resumption

**Log persistence:**
- Stdout and stderr are captured continuously to `experiments/logs/<experiment_id>.log`
- Log file is opened at run start; closed (and flushed) at run end
- Log is retained for all lifecycle states including `failed` and `interrupted`
- Log includes: config hash confirmation, epoch-level summaries, and full exception trace on failure

---

## 6. NAS Compatibility

NAS trial generation from Q32 (classical), Q33 (quantum), and Q34 (multi-objective pilot) must produce configs that conform to this schema without modification to the runner.

**How NAS trial generation works:**
1. The search algorithm generates a `model.parameters` dict for each trial
2. The dict is injected into a base config template for the search phase
3. The `experiment.phase` is set to the NAS phase identifier (`classical_nas`, `quantum_nas_dv`, etc.)
4. The `experiment_id` is auto-generated at run start; all other immutable fields are set by the template
5. The runner executes the trial config identically to any hand-crafted config

**Classical NAS (Q32)** varies parameters within `model.parameters`:
```yaml
model:
  architecture: classical_nas_trial
  backbone: null              # or frozen pretrained backbone
  backbone_frozen: true
  head: ClassicalNASHead
  parameters:
    block_type: depthwise_sep  # varied: standard | depthwise_sep | inverted_residual
    channels: [64, 128]        # varied: [32,64] | [64,128] | [64,128,256] etc.
    depth: 2                   # varied: 1 | 2 | 3
    pooling: adaptive_avg      # varied: adaptive_avg | max | gap
    dropout: 0.3               # varied: 0.0 | 0.1 | 0.3
```

**DV Quantum NAS (Q33)** varies quantum head parameters:
```yaml
model:
  architecture: dv_hybrid
  backbone: c006_d040_classical_anchor
  backbone_frozen: true
  head: DVQuantumHead
  parameters:
    n_qubits: 4               # varied: 2 | 4 | 6 | 8
    circuit_depth: 1          # varied: 1 | 2 | 3
    ansatz_type: rotational   # varied: rotational | hardware_efficient | strongly_entangling
    compression_dim: 4        # varied: 2 | 4 | 8
```

**CV Quantum NAS (Q33)** varies Gaussian head parameters:
```yaml
model:
  architecture: cv_hybrid
  backbone: c006_d040_classical_anchor
  backbone_frozen: true
  head: CVGaussianHead
  parameters:
    n_modes: 2               # varied: 1 | 2 | 4
    cv_depth: 1              # varied: 1 | 2 | 3
    squeezing_cap: 1.5       # varied: 1.0 | 1.5 | 2.0
    hbar: 2.0                # fixed
    compression_dim: 4       # varied: 2 | 4 | 8
    encoding_scheme: displacement  # varied: displacement | squeezed_displacement
```

The runner does not inspect `model.parameters` beyond forwarding it to the model factory. No runner modification is required to support new search dimensions.

---

## 7. Reproducibility Requirements

**Seed initialization sequence.** The runner sets all random seeds in the following order before any other operation:
1. `random.seed(config.reproducibility.seed)`
2. `numpy.random.seed(config.reproducibility.seed)`
3. `torch.manual_seed(config.reproducibility.seed)`
4. `torch.cuda.manual_seed_all(config.reproducibility.seed)` if CUDA available
5. `torch.backends.cudnn.deterministic = True`
6. `torch.backends.cudnn.benchmark = False`

This sequence must execute before dataset loading, model initialization, and any data augmentation.

**Git commit recording.** At run start, the runner executes:
```bash
git rev-parse HEAD
git diff --quiet || echo dirty
```
The result is stored as `reproducibility.git_commit` in the experiment metadata. If the working tree is dirty, the value is `<SHA>-dirty` and a warning is logged. The experiment proceeds but the dirty flag is permanent.

**Hardware metadata.** The runner auto-populates `reproducibility.hardware` at run start:
- `gpu_model`: `torch.cuda.get_device_name(0)` if available, else `cpu`
- `cuda_version`: `torch.version.cuda` if available, else `null`
- `cpu_fallback`: `not torch.cuda.is_available()`
- `gpu_memory_mb`: `torch.cuda.get_device_properties(0).total_memory // (1024**2)` if available

**Reproducibility guarantee.** Given a frozen config and a matching `git_commit` SHA, the experiment must produce identical results (within numerical tolerance of float32) when re-run. The tolerance is defined as: AUROC within ±0.0001, F1 within ±0.0001. Any experiment that fails this criterion has a hidden source of non-determinism that must be diagnosed and corrected.

---

## 8. Command Block (Formalized in Q31A)

The `command` block is a required top-level section in all runner-executed experiment configs. It was introduced in Q31 and formalized as a first-class required schema field in Q31A.

```yaml
command:
  executable: python3       # required
  args:                     # required list of strings
    - scripts/my_script.py
    - --arg1
    - value1
```

### Purpose

The `command` block decouples the runner from model-specific logic. The runner does not inspect or interpret the command; it executes `[executable] + args` as a subprocess, captures stdout/stderr, and records the return code. This design means:

- The runner can wrap any script (smoke tests, training scripts, evaluation scripts, NAS trial executors) without modification
- NAS trial generation (Q32+) produces valid `command` blocks programmatically; the runner executes them identically to hand-crafted configs
- Model and ansatz code changes never require runner changes

### Required Fields

| Field | Required | Type | Notes |
|---|---|---|---|
| `command.executable` | Yes | string | Executable name or path; use `python3` for containers |
| `command.args` | Yes | list of strings | Script path + arguments; all values must be strings |

### NAS Compatibility

NAS trial generation (Q32, Q33, Q34) populates `command.args` with the appropriate training script and the trial-specific hyperparameters passed as CLI arguments. The runner does not need modification to execute NAS-generated commands.

**Example NAS trial command block:**
```yaml
command:
  executable: python3
  args:
    - scripts/train_vindr_cv_binary.py
    - --n-modes
    - "4"
    - --cv-depth
    - "2"
    - --seed
    - "42"
```

### Container Compatibility Note

The `docker-qstrata-gpu` container has `python3` in PATH but not `python`. All experiment configs targeting this container must use `executable: python3`.

---

## 9. Scientific Integrity Requirements

**All hyperparameters that affect results must appear in the config.** There are no environment-variable-only overrides. There are no implicit defaults that differ from the documented schema defaults. If a value affects any metric in any way, it must be in `model.parameters` or another tracked field.

**Leaderboard entries cite `experiment_id` only.** Comparative statements such as "model A outperforms model B" are valid only when they cite the `experiment_id` values of both experiments. `experiment_id` references are stable and machine-verifiable. Free-form descriptions are not.

**Reproducibility testing.** Before any NAS search begins in Q32, the runner must pass a reproducibility test: run the same config twice with the same seed and confirm results match within the defined tolerance. This test is documented in the Q31 runner implementation.

**No implicit hyperparameter defaults.** Optional fields that are not provided in a config take the values documented in this schema's field definitions. The runner must not apply any additional defaults beyond what is documented here. If a new default is introduced, it must be added to this schema document and all affected existing experiment records must be noted.

---

```
Schema version: 1.1 (Q31A — command block formalized)
Implemented: Q31
Hardened: Q31A (command block, env var git fallback)
NAS compatibility: Q32 (classical), Q33 (DV/CV quantum), Q34 (multi-objective)
```
