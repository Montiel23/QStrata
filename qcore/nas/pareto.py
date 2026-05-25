"""
qcore/nas/pareto.py

NAS v1 Pareto ranking — Slice 25.

Computes Pareto dominance across three objectives for a list of
EvaluatorResult objects and produces a ranked markdown summary table.

Objectives (fixed for v1):
  - Maximize  best_val_acc
  - Minimize  params
  - Minimize  latency_ms

test_acc_analysis_only must never appear in any dominance or ranking
calculation in this module.

Public API
----------
    pareto_dominates(a, b)              -> bool
    compute_pareto_front(results)       -> list[str]
    rank_candidates(results)            -> list[EvaluatorResult]
    render_markdown_table(results, pareto_front) -> str
"""

from __future__ import annotations

from typing import List

# EvaluatorResult imported at the call site; accepted here as typed hints only
# to avoid a hard coupling at module load time.  The actual type check is
# structural (duck-typed), so the module works even if EvaluatorResult is
# passed as a plain object with the required attributes.
#
# We do import it for the type annotation in the public signature.
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qcore.nas.evaluator import EvaluatorResult


# ── Pareto dominance ──────────────────────────────────────────────────────────

def pareto_dominates(a: EvaluatorResult, b: EvaluatorResult) -> bool:
    """
    Return True if candidate *a* Pareto-dominates candidate *b*.

    Dominance is defined across three objectives only:
      - best_val_acc  (maximise)
      - params        (minimise)
      - latency_ms    (minimise)

    *a* dominates *b* when:
      1. *a* is at least as good as *b* on every objective, AND
      2. *a* is strictly better than *b* on at least one objective.

    Only the three v1 fitness objectives are evaluated here; the analysis-only
    accuracy field is never referenced in this function.

    Parameters
    ----------
    a, b : EvaluatorResult
        Two candidates to compare.

    Returns
    -------
    bool
        True if *a* dominates *b*; False otherwise.
    """
    # At-least-as-good on all three objectives
    at_least_as_good = (
        a.best_val_acc >= b.best_val_acc   # higher is better
        and a.params       <= b.params     # lower is better
        and a.latency_ms   <= b.latency_ms # lower is better
    )
    if not at_least_as_good:
        return False

    # Strictly better on at least one objective
    strictly_better = (
        a.best_val_acc > b.best_val_acc
        or a.params       < b.params
        or a.latency_ms   < b.latency_ms
    )
    return strictly_better


# ── Pareto front ──────────────────────────────────────────────────────────────

def compute_pareto_front(results: List[EvaluatorResult]) -> List[str]:
    """
    Identify the Pareto front from a list of evaluated candidates.

    A candidate is on the Pareto front if no other candidate dominates it
    across the three v1 objectives (best_val_acc, params, latency_ms).

    Parameters
    ----------
    results : list[EvaluatorResult]
        Non-empty list of evaluated candidates.

    Returns
    -------
    list[str]
        Candidate IDs on the Pareto front, in the order they first appear
        in *results*.

    Raises
    ------
    ValueError
        If *results* is empty or None.
    """
    if not results:
        raise ValueError(
            "compute_pareto_front: results list is empty — "
            "at least one EvaluatorResult is required."
        )

    front = []
    for candidate in results:
        dominated = any(
            pareto_dominates(other, candidate)
            for other in results
            if other.candidate_id != candidate.candidate_id
        )
        if not dominated:
            front.append(candidate.candidate_id)
    return front


# ── Ranking ───────────────────────────────────────────────────────────────────

def rank_candidates(results: List[EvaluatorResult]) -> List[EvaluatorResult]:
    """
    Return all candidates ordered by Pareto rank, then by best_val_acc.

    Ordering rules:
      1. Pareto-front candidates come first (not dominated by any other).
      2. Dominated candidates follow.
      3. Within each group, candidates are sorted by best_val_acc descending
         (highest accuracy first).

    Parameters
    ----------
    results : list[EvaluatorResult]
        Non-empty list of evaluated candidates.

    Returns
    -------
    list[EvaluatorResult]
        All candidates in ranked order.

    Raises
    ------
    ValueError
        If *results* is empty or None.
    """
    if not results:
        raise ValueError(
            "rank_candidates: results list is empty — "
            "at least one EvaluatorResult is required."
        )

    pareto_front = set(compute_pareto_front(results))

    front_group     = [r for r in results if r.candidate_id in pareto_front]
    dominated_group = [r for r in results if r.candidate_id not in pareto_front]

    front_group.sort(    key=lambda r: r.best_val_acc, reverse=True)
    dominated_group.sort(key=lambda r: r.best_val_acc, reverse=True)

    return front_group + dominated_group


# ── Markdown table ────────────────────────────────────────────────────────────

def render_markdown_table(
    results:      List[EvaluatorResult],
    pareto_front: List[str],
) -> str:
    """
    Render a markdown summary table for all evaluated candidates.

    Columns:
      Rank | Candidate | Block type | Channels | Params | Best val acc |
      Final train acc | Test acc* | Latency (ms/batch) | Pareto

    The table is ordered by rank (Pareto-front first, then dominated),
    with best_val_acc descending within each group.

    test_acc_analysis_only is included in the table with a clear footnote
    indicating it is for analysis only and must not be used as a fitness
    signal.

    Parameters
    ----------
    results : list[EvaluatorResult]
        Non-empty list of evaluated candidates.
    pareto_front : list[str]
        Candidate IDs on the Pareto front (from compute_pareto_front).

    Returns
    -------
    str
        Rendered markdown table string (includes trailing footnote).

    Raises
    ------
    ValueError
        If *results* is empty or None.
    """
    if not results:
        raise ValueError(
            "render_markdown_table: results list is empty — "
            "at least one EvaluatorResult is required."
        )

    front_set = set(pareto_front)
    ranked    = rank_candidates(results)

    header = (
        "| Rank | Candidate | Block type | Channels | Params | "
        "Best val acc | Final train acc | Test acc\\* | "
        "Latency (ms/batch) | Pareto |"
    )
    separator = (
        "|---:|---|---|---|---:|---:|---:|---:|---:|:---:|"
    )

    rows = []
    for rank, r in enumerate(ranked, start=1):
        pareto_marker = "✓" if r.candidate_id in front_set else ""
        row = (
            f"| {rank} "
            f"| {r.candidate_id} "
            f"| {r.block_type} "
            f"| {r.conv_channels} "
            f"| {r.params:,} "
            f"| {r.best_val_acc:.2f}% "
            f"| {r.final_train_acc:.2f}% "
            f"| {r.test_acc_analysis_only:.2f}% "
            f"| {r.latency_ms:.3f} "
            f"| {pareto_marker} |"
        )
        rows.append(row)

    footnote = (
        "\n> \\*Test accuracy is reported for analysis only and must not be "
        "used as a fitness signal."
    )

    return "\n".join([header, separator] + rows) + footnote
