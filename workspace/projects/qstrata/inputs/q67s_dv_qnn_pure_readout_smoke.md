# Q67S — DV-QNN Pure Readout Smoke

**Slice ID**: Q67S-DV-QNN-PURE-READOUT-SMOKE  
**Campaign**: pure_quantum_readout_smoke  
**Status**: BLOCKED  
**Depends on**: Q66 (READY)  
**Estimated runtime**: LOW (< 5 min for 1 epoch / 64 samples)  
**Date planned**: After Q66 completes  

---

## 1. Objective

Smoke-test three pure Born-rule readout variants for DV-QNN. Replace the classical
`Linear(16→2)` readout with a measurement-based class score. Validate that all three
variants run, produce valid AUROC, and log correct quantum parameter counts.

Constraints:
- max 64 train samples / 32 val samples / 32 test samples
- 1 seed: 42
- 1 epoch maximum
- VinDr-SpineXR only
- No classical Linear(n→2) after quantum measurement
- Custom qcore framework only

---

## 2. Architecture

### Shared components

| Component | Type | Params | Notes |
|-----------|------|--------|-------|
| Encoder Linear(128→4) | Classical, trainable | 516 | Compresses Q49 embedding to n_qubits=4 |
| medical_ansatz (4 qubits, depth=1) | Quantum circuit | 0 (no own params) | H+reupload+variational+CNOT |
| theta parameters | Quantum, trainable | 24 | Shape (1, 2, 4, 3) |
| DV backend (compile+run) | Quantum | 0 | 16×16 unitary; Born rule |

### Readout Variant A — Parity Readout

```python
# After Born-rule: probs ∈ R^16
# Assign basis state i to class based on popcount parity
# class=1 states: those where bin(i).count('1') % 2 == 1  (odd parity)
# class=0 states: those where bin(i).count('1') % 2 == 0  (even parity)
p_class1 = sum(probs[i] for i in range(16) if bin(i).count('1') % 2 == 1)
p_class0 = 1.0 - p_class1
score = p_class1  # → AUROC
```

No trainable params after measurement. Total trainable: 516 + 24 = 540.

### Readout Variant B — Top-k Probability Mass

```python
# On a single forward pass of the training set (no gradient):
#   for each sample, record (probs, label)
#   compute mean probs conditioned on class: mu_probs_c0, mu_probs_c1
#   assign basis state i to class c = argmax([mu_probs_c0[i], mu_probs_c1[i]])
#   freeze this assignment for the training/eval loop
# During training and eval:
p_class1 = sum(probs[i] for i in class1_states)  # class1_states from above
score = p_class1
```

State assignment is learned once before training (1 forward pass, no gradient).
Not a trainable parameter — it is a fixed lookup table derived from class-conditioned means.
Total trainable: 516 + 24 = 540.

### Readout Variant C — Expectation Value

```python
# Z-expectation on qubit 0 (first qubit's Z operator)
# <Z_0> = sum_i (-1)^(bit_0(i)) * probs[i]
# bit_0(i) = (i >> (n_qubits-1)) & 1  (most-significant qubit)
score = sum((-1)**(( i >> (n_qubits-1)) & 1) * probs[i] for i in range(16))
# score ∈ [-1, 1]; threshold at 0: class = 1 if score > 0
# For AUROC: use (score + 1) / 2 as probability score
```

Optional calibration: single trainable `nn.Parameter` scalar bias initialized to 0.
If calibration used: report explicitly and ablate (frozen vs trained).
Total trainable (no calibration): 516 + 24 = 540.

---

## 3. Protocol

- Load first 64 train samples + 32 val samples from Q49 embeddings.
- Train for 1 epoch with Adam (lr=1e-3, wd=1e-4), batch_size=8.
- Evaluate on 32 val samples.
- Repeat for all 3 variants.
- Log: AUROC, F1, accuracy, n_trainable_params, n_quantum_params, readout_type, runtime_s.

---

## 4. Quantum State Logging (required)

For each variant, log from the val set:
- `per_sample_probs`: shape (N_val, 16) — Born-rule probability vectors
- `measurement_entropy`: shape (N_val,) — `S = -sum(p * log2(p+eps))`
- `probability_margin`: shape (N_val,) — `max(p_class1_states) - max(p_class0_states)`
- `class_conditioned_mean_probs`: shape (2, 16) — mean probs per class

---

## 5. Outputs

| File | Description |
|------|-------------|
| `workspace/experiments/Q67S/results/q67s_smoke_results.json` | Full results dict |
| `workspace/experiments/Q67S/results/q67s_smoke_metrics.csv` | Per-variant metrics table |
| `workspace/experiments/Q67S/reports/q67s_dv_qnn_pure_readout_smoke_report.md` | Analysis |
| `reports/q67s_dv_qnn_pure_readout_smoke.md` | Publication copy |

---

## 6. Pass Criteria

- [ ] All 3 readout variants run without error
- [ ] AUROC, F1, accuracy logged per variant
- [ ] `n_quantum_params = 24` confirmed for all variants
- [ ] No `nn.Linear(16, 2)` or similar classical readout layer after measurement
- [ ] Measurement entropy logged for val set
- [ ] Delta AUROC vs Q57 hybrid (0.8842) documented
- [ ] Runtime < 30 min
---

## Mode

analysis

## Validation Commands

- sliceforge campaign validate --project qstrata
