# Q46G — Overnight Autonomy Pre-flight

**Slice ID**: Q46G-OVERNIGHT-AUTONOMY-PREFLIGHT
**Date**: 2026-06-01
**Author**: Claude Code (Sonnet 4.6)
**Branch**: feature/q46g_overnight_autonomy_preflight
**Scope**: Synthesize Q46C §5 (runtime contract) and Q46E (execution environment audit) into a
single go/no-go authorisation document for running the Q46B feature extractor benchmark
unattended overnight.

---

## Verdict

> **CONDITIONAL GO**
>
> All hard infrastructure preconditions are satisfied. Two gate items must be cleared before
> launching unattended:
> 1. Pre-download ImageNet weights inside the container (FM-14 mitigation).
> 2. Downgrade NumPy to `<2` inside the container (FM-15 mitigation).
>
> With both gates cleared, the benchmark may be launched unattended. Expected total wall time
> for Phase 1 + Phase 2 combined is well under 3 hours. No human intervention should be
> required after launch.

---

## 1. Source Documents

| Ref | Document | Date | Status |
|---|---|---|---|
| Q46C | `reports/q46c_feature_extractor_benchmark_scaffold.md` (§5 — Runtime Contract) | 2026-05-31 | COMPLETE |
| Q46E | `reports/q46e_execution_environment_audit.md` | 2026-06-01 | COMPLETE |
| Q47 | `docs/specs/q46_benchmark_failure_recovery_playbook.md` | 2026-06-01 | COMPLETE |

No Q46D standalone report exists; the runtime contract is captured in Q46C §5 and is reproduced
in Section 3 below.

---

## 2. Environment Readiness Summary (Q46E)

| Domain | Status | Detail |
|---|---|---|
| Dataset | ✅ READY | 10,466 images — train/val/test × class 0/1 all populated |
| Baseline checkpoint | ✅ READY | `checkpoints/c006_d040_classical_anchor.pt` (52 KB) |
| Head config | ✅ READY | `configs/experiments/q34a_classical_nas/q34a_trial_004.yaml` |
| Benchmark script | ✅ READY | `scripts/run_q46b_feature_extractor_benchmark.py` |
| Benchmark config | ✅ READY | `configs/q46_feature_extractor_benchmark.yaml` |
| Docker container | ✅ READY | `docker-qstrata-gpu-1` running, Up 47 h, GPU-bound (`nvidia`) |
| PyTorch / torchvision | ✅ READY | torch 2.2.2+cu121, torchvision 0.17.2+cu121 — importable, CUDA active |
| GPU hardware | ✅ READY | RTX 2060 SUPER — 8 GB VRAM, ~7.4 GB free (3% utilisation) |
| Output directories | ✅ READY | `experiments/leaderboards/` and `experiments/results/` exist |
| NumPy ABI | ⚠️ ADVISORY | numpy 2.2.6 violates Dockerfile `numpy<2` constraint; torch imports with UserWarning but operates correctly |
| ImageNet weights (cache) | ⚠️ GATE ITEM | `~/.cache/torch/hub/checkpoints/` not verified as populated — risk of overnight download failure (FM-14) |

**Q46E overall classification: READY_WITH_WARNINGS**

---

## 3. Runtime Contract (Q46C §5)

### 3.1 Phase Definitions

| Phase | Seeds | Candidates | Epochs | Wall-Time Cap | Output |
|---|---|---|---|---|---|
| Phase 1 — Smoke | [42] | All 5 | 4 | 60 min | `experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv` |
| Phase 2 — Full | [42, 7, 123] | All 5 | 4 | 120 min | `experiments/leaderboards/q46b_extractor_full_leaderboard.csv` |
| Phase 3 — Extended | [42, 7, 123, 999, 2025] | Winner only | 4 | 60 min | `experiments/leaderboards/q46b_extractor_extended_leaderboard.csv` |

Phase 3 is conditional: triggered only if Phase 2 winner's mean AUROC is within 0.005 of the
Q38C ceiling (0.7239). If the winner exceeds 0.7284, Phase 3 is not triggered.

### 3.2 Decision Rule

```
WINNER   : candidate mean_auroc > 0.7196 (Q45A) AND seeds_beat_q45a >= 2/3
NEGATIVE : no candidate meets both conditions
EXTENDED : Phase 2 winner mean_auroc within 0.005 of Q38C ceiling (0.7239) → run Phase 3
```

### 3.3 Wall-Time Estimates (from Q46E §6, Q46C scaffold)

| Phase | Candidates | Seeds | Max epoch time | Estimated total |
|---|---|---|---|---|
| Phase 1 — Smoke | 5 | 1 | ~35 s/candidate (mobilenetv3) to ~70 s (convnext_tiny) | ~5 min |
| Phase 2 — Full | 5 | 3 | Same per-seed rates | ~25 min |
| Phase 3 — Extended | 1 (winner) | 5 | ~35 s/seed | ~3 min |

All three phases combined are estimated at < 35 min total. The 60 min (Phase 1) and 120 min
(Phase 2) caps are conservative safety rails — no cap is expected to fire under normal
conditions.

### 3.4 Execution Commands

```bash
# Pre-flight dry-run (always run first):
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --dry-run

# Phase 1 — Smoke:
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --smoke

# Phase 2 — Full (only if Phase 1 produces ≥1 candidate meeting WINNER conditions):
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --full
```

---

## 4. Overnight Risk Register

### 4.1 Blocking Risks

| Risk | FM Ref | Likelihood | Pre-flight Mitigation |
|---|---|---|---|
| Container exits during the run | FM-01 | Low-Medium (desktop GPU; container Up 47h — not guaranteed overnight) | Verify container is still running immediately before launch. Use `docker start` if needed. |
| ImageNet weights not cached → download failure | FM-14 | **HIGH** | **GATE ITEM: pre-download all 4 torchvision weights before launching (see §5)** |
| GPU becomes unavailable (suspend/resume, driver update) | FM-06 | Low | Check `nvidia-smi` before launch; confirm `torch.cuda.is_available()` in container |

### 4.2 Non-Blocking / Advisory Risks

| Risk | FM Ref | Likelihood | Pre-flight Mitigation |
|---|---|---|---|
| NumPy ABI UserWarning on torch import | FM-15 | **Certain** (numpy 2.2.6 present) | **GATE ITEM: `pip install "numpy<2" --force-reinstall` in container before launch** |
| Wall-time cap fires (ConvNeXt-Tiny slowest) | FM-07 | Low | ConvNeXt-Tiny estimated ~70 s/seed × 3 seeds = ~3.5 min. Cap of 120 min is ample. |
| CUDA OOM (desktop processes consume VRAM) | FM-08 | Very Low | ~7.4 GB free at audit time; batch_size=8 × 224×224 well within 8 GB |
| Host RAM OOM (persistent DataLoader workers) | FM-10 | Very Low | 4 workers × 5 candidates managed sequentially; RAM expected adequate |
| Results JSON absent at completion | FM-16 | Certain (not implemented in Q46C scaffold) | Expected — CSV is the authoritative output; JSON can be generated post-hoc |
| Script invoked outside Docker | FM-02 | Very Low | Use `docker exec` prefix; host `.venv/` (Python 3.13) intentionally has no torch |

### 4.3 Risk Severity Map

```
BLOCKING (must resolve before launch):
  FM-14 — ImageNet weights not pre-cached  ← PRIMARY GATE ITEM
  FM-01 — Container not running at launch time

ADVISORY (resolve before launch, non-fatal if skipped):
  FM-15 — NumPy 2.x ABI warning            ← SECONDARY GATE ITEM

NON-BLOCKING (expected or recoverable):
  FM-07 — Wall-time cap (very unlikely to fire)
  FM-08 — VRAM OOM (very unlikely at batch_size=8)
  FM-10 — Host RAM OOM (very unlikely)
  FM-16 — JSON absent (by design in Q46C scaffold)
```

---

## 5. Gate Items — Must Clear Before Overnight Launch

The following five commands must all pass. Execute in order immediately before starting the
unattended run.

```bash
# GATE 1: Container running and GPU accessible
docker exec docker-qstrata-gpu-1 \
    python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: CUDA: True NVIDIA GeForce RTX 2060 SUPER

# GATE 2: Pre-download ImageNet weights (FM-14 mitigation — critical for unattended run)
docker exec docker-qstrata-gpu-1 python3 - <<'EOF'
from torchvision.models import efficientnet_b0, mobilenet_v3_small, mobilenet_v3_large, convnext_tiny
for fn in [efficientnet_b0, mobilenet_v3_small, mobilenet_v3_large, convnext_tiny]:
    m = fn(weights="IMAGENET1K_V1")
    print(f"Cached: {fn.__name__}")
    del m
print("All ImageNet weights cached.")
EOF

# GATE 3: NumPy ABI fix (FM-15 mitigation — advisory but strongly recommended)
docker exec docker-qstrata-gpu-1 pip install "numpy<2" --force-reinstall -q
docker exec docker-qstrata-gpu-1 python3 -c "import torch; print('NumPy ABI: OK')" 2>&1 | grep -v UserWarning

# GATE 4: VRAM headroom (require ≥ 2 GB free)
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
# Expected: used < 6144, free > 2048

# GATE 5: Dry-run passes
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --dry-run
# Expected: DRY-RUN RESULT: PASS
```

All five gates must show the expected output before proceeding. If Gate 1 or Gate 5 fails,
do not launch unattended.

---

## 6. Pre-flight Checklist

### 6.1 Infrastructure Checks (Q46E Source)

- [ ] `docker ps | grep docker-qstrata-gpu-1` shows container RUNNING
- [ ] `torch.cuda.is_available()` returns `True` inside container
- [ ] `torch.cuda.get_device_name(0)` returns `NVIDIA GeForce RTX 2060 SUPER`
- [ ] `nvidia-smi` shows VRAM free ≥ 2,048 MiB
- [ ] Dataset root `/workspace/data/processed/vindr_binary_roi_224/{train,val,test}/{0,1}` all populated
- [ ] Baseline checkpoint `/workspace/checkpoints/c006_d040_classical_anchor.pt` present (~52 KB)
- [ ] Head config `/workspace/configs/experiments/q34a_classical_nas/q34a_trial_004.yaml` present
- [ ] Output dirs `experiments/leaderboards/` and `experiments/results/` exist

### 6.2 Gate Items (Must Be Active, Not Just Checked)

- [ ] **GATE 2 DONE**: All 4 torchvision backbone weights confirmed downloaded/cached
- [ ] **GATE 3 DONE**: `numpy<2` reinstalled; torch import shows no UserWarning
- [ ] **GATE 5 DONE**: Dry-run returns `DRY-RUN RESULT: PASS`

### 6.3 Execution Sequence

- [ ] Phase 1 (`--smoke`) launched
- [ ] Phase 1 leaderboard CSV written: `experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv`
- [ ] Phase 2 (`--full`) launched (conditional on Phase 1 producing ≥1 WINNER-eligible candidate)
- [ ] Phase 2 leaderboard CSV written: `experiments/leaderboards/q46b_extractor_full_leaderboard.csv`
- [ ] Decision rule evaluated (see FM-17 reference in playbook for correct 2-condition evaluation)

---

## 7. Post-Run Triage Quick Reference

If the run exits abnormally, consult the failure recovery playbook
(`docs/specs/q46_benchmark_failure_recovery_playbook.md`) for full procedures.

| Symptom | Most Likely FM | First Command |
|---|---|---|
| No container / daemon error | FM-01 | `docker start docker-qstrata-gpu-1` |
| `No module named 'torch'` | FM-02 | Re-run with `docker exec` prefix |
| `No module named 'qcore'` | FM-03 | `docker compose -f docker-compose.gpu.yml up -d` |
| CSV empty or absent after run | FM-11 | Check stdout for `ERROR building backbone` lines |
| `[WALL-TIME CAP]` in log | FM-07 | Re-run with `--candidate <missing_id>` and merge CSVs |
| `OutOfMemoryError: CUDA` | FM-08 | `nvidia-smi`; kill GPU competitors; re-run |
| AUROC = 0.5 for all runs | FM-13 | Check trainable param count; verify device consistency |
| `ConnectionError` for weights | FM-14 | Gate 2 was not completed; pre-download in container and re-run |
| NumPy UserWarning on import | FM-15 | `pip install "numpy<2" --force-reinstall` in container |
| JSON absent after completion | FM-16 | Expected — use FM-16 manual generation snippet |

---

## 8. Go/No-Go Classification

| Criterion | Result |
|---|---|
| All Q46E hard preconditions satisfied | ✅ YES |
| Runtime contract understood and caps adequate | ✅ YES |
| Primary overnight failure risk identified and mitigated | ✅ FM-14 (ImageNet weights — Gate 2) |
| Secondary risk identified and advisory action defined | ✅ FM-15 (NumPy ABI — Gate 3) |
| Failure recovery playbook available for all FM codes | ✅ Q47 playbook covers FM-01 through FM-17 |
| Unresolved hard blockers remaining | ✅ NONE |

**FINAL VERDICT: CONDITIONAL GO**

Launch is authorised once Gate 2 (ImageNet weights cached), Gate 3 (NumPy ABI fixed), and
Gate 5 (dry-run passes) are confirmed. With those three items verified, the Q46B benchmark
can run unattended overnight with no expected intervention required.

Estimated total overnight wall time: **< 35 minutes** (Phase 1 + Phase 2 combined).
Wall-time caps (60 min / 120 min) will not be challenged under normal desktop-idle conditions.

---

## 9. Validation

| Check | Status |
|---|---|
| Pre-flight checklist produced | ✅ Section 6 |
| GO / NO-GO / CONDITIONAL classification assigned | ✅ CONDITIONAL GO — Section 8 |
| No benchmark executed | ✅ Planning-only slice |
| Report at `reports/q46g_overnight_autonomy_preflight.md` | ✅ This document |
| Sources: Q46C §5 (runtime contract) + Q46E (environment audit) | ✅ Sections 2–3 |
