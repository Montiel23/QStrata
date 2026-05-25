        # NAS v1 Evaluation Report

        - **Protocol:** nas_benchmark_protocol_v1 — seed 42, best-val checkpoint selection
        - **Candidates:** C001, C006, C004
        - **Objectives:** maximize best_val_acc, minimize params, minimize latency_ms

        ---

        ## Results

        | Rank | Candidate | Block type | Channels | Params | Best val acc | Final train acc | Test acc\* | Latency (ms/batch) | Pareto |
|---:|---|---|---|---:|---:|---:|---:|---:|:---:|
| 1 | C001 | standard | [32, 64] | 19,138 | 92.56% | 90.46% | 86.54% | 0.590 | ✓ |
| 2 | C006 | depthwise_sep | [64, 128] | 9,870 | 91.98% | 91.53% | 86.22% | 0.589 | ✓ |
| 3 | C004 | depthwise_sep | [32, 64] | 2,894 | 91.41% | 89.95% | 87.66% | 0.698 | ✓ |
> \*Test accuracy is reported for analysis only and must not be used as a fitness signal.

        ---

        ## Pareto Front

        The following candidates are not dominated by any other on all three objectives:

        - **C001** — standard [32, 64], best val acc 92.56%, 19,138 params, 0.590 ms/batch
- **C004** — depthwise_sep [32, 64], best val acc 91.41%, 2,894 params, 0.698 ms/batch
- **C006** — depthwise_sep [64, 128], best val acc 91.98%, 9,870 params, 0.589 ms/batch

        ---

        ## Notes

        - test_acc_analysis_only is reported in the table for post-hoc analysis only.
          It must not be used as a fitness signal or in Pareto dominance calculations.
        - Evaluation order: C001, C006, C004 (sequential, single GPU).
        - No dashboards, no databases, no tracking servers, no MLflow, no monitoring stack.
