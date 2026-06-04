# Q73 — Noise Detector Circuit Ablation

**Slice ID**: Q73-NOISE-DETECTOR-CIRCUIT-ABLATION  
**Campaign**: pure_quantum_readout_full  
**Status**: BLOCKED  
**Depends on**: Q72  
**Estimated runtime**: HIGH (many inference sweeps; no re-training for noise ablation; re-train for squeezing_cap sweep)  

---

## 1. Objective

Test robustness of pure quantum readout models to realistic noise and sweep circuit
hyperparameters. Noise ablations apply at inference time (no re-training). 
Squeezing cap sweep requires re-training with each cap value.

---

## 2. Ablation Sweeps

### DV-QNN Noise Ablations (inference-only, models from Q69)

| Sweep | Parameter | Values | Notes |
|-------|-----------|--------|-------|
| theta_noise | Gaussian noise N(0,σ²) added to theta at inference | σ ∈ {0.0, 0.01, 0.05, 0.1, 0.2} | Simulates hardware theta imprecision |
| input_noise | N(0,σ²) added to encoder output x_i | σ ∈ {0.0, 0.01, 0.05, 0.1} | Simulates encoding noise |

### CV-QNN Noise Ablations (inference-only, models from Q70)

| Sweep | Parameter | Values | Notes |
|-------|-----------|--------|-------|
| electronic_noise | N(0,σ_e²) added to homodyne output mu[::2] | σ_e ∈ {0.0, 0.01, 0.05, 0.1, 0.5} | Simulates real detector noise |
| detector_efficiency | Scale homodyne output by η: mu[::2] *= η | η ∈ {0.5, 0.7, 0.9, 1.0} | Simulates detection loss |

### CV-QNN Circuit Ablations (require re-training)

| Sweep | Parameter | Values |
|-------|-----------|--------|
| squeezing_cap_sweep | squeezing_cap in GaussianVariationalAnsatz | {0.5, 1.0, 1.5, 2.0, 2.5} |

Run squeezing cap sweep: 1 seed (42), VinDr-SpineXR only, 4 epochs.

---

## 3. Metrics per Sweep

For each sweep point: AUROC, F1, accuracy. Report as sweep curve (x=noise param, y=AUROC).

---

## 4. Outputs

| File | Description |
|------|-------------|
| `workspace/experiments/Q73/results/dv_qnn_theta_noise_sweep.csv` | |
| `workspace/experiments/Q73/results/dv_qnn_input_noise_sweep.csv` | |
| `workspace/experiments/Q73/results/cv_qnn_electronic_noise_sweep.csv` | |
| `workspace/experiments/Q73/results/cv_qnn_detector_efficiency_sweep.csv` | |
| `workspace/experiments/Q73/results/cv_qnn_squeezing_cap_sweep.csv` | |
| `workspace/experiments/Q73/figures/noise_robustness_curves.svg` | AUROC vs noise level |
| `workspace/experiments/Q73/figures/squeezing_cap_auroc.svg` | AUROC vs squeezing cap |
| `workspace/experiments/Q73/figures/detector_efficiency_auroc.svg` | AUROC vs η |
| `workspace/experiments/Q73/reports/q73_noise_detector_ablation_report.md` | |
| `reports/q73_noise_detector_circuit_ablation.md` | |

---

## 5. Pass Criteria

- [ ] DV theta noise sweep (5 values) completed
- [ ] CV electronic noise sweep (5 values) completed
- [ ] CV detector efficiency sweep (4 values) completed
- [ ] CV squeezing cap sweep (5 values) completed (re-training allowed)
- [ ] All 3 figures generated
- [ ] Noise robustness documented: AUROC at σ=0.1 vs σ=0.0
---

## Mode

analysis

## Validation Commands

- sliceforge campaign validate --project qstrata
