"""
qcore/experiments/leaderboard.py

Leaderboard CSV management for the QStrata experiment runner (Q31 MVP).

One CSV file per phase, located at experiments/leaderboards/<phase>.csv.
Each completed or failed experiment appends one row to its phase leaderboard.

Uses Python's built-in csv module only. No pandas dependency.
"""

from __future__ import annotations

import csv
import os

# Canonical column order for all phase leaderboard CSVs.
# All experiments write the same columns regardless of phase.
LEADERBOARD_COLUMNS: list[str] = [
    "experiment_id",
    "timestamp_start",
    "phase",
    "status",
    "smoke_pass",
    "return_code",
    "duration_seconds",
    "git_commit",
    "git_dirty",
]


def update_leaderboard(leaderboard_path: str, row_dict: dict) -> None:
    """
    Append one experiment row to the phase leaderboard CSV.

    Creates the file and writes the header row if it does not exist.
    Appends a data row on every call. Does not sort or deduplicate —
    the leaderboard is an append-only log; ranking is done at read time.

    Args:
        leaderboard_path: Absolute or relative path to the phase CSV file.
        row_dict:         Dict with at minimum the LEADERBOARD_COLUMNS keys.
                          Extra keys are silently ignored (extrasaction="ignore").
                          Missing keys are written as empty strings.
    """
    parent = os.path.dirname(leaderboard_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    file_exists = os.path.isfile(leaderboard_path)

    with open(leaderboard_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=LEADERBOARD_COLUMNS,
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row_dict.get(col, "") for col in LEADERBOARD_COLUMNS})
