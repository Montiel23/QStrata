"""
qcore/experiments/schema.py

Config schema validation for the QStrata experiment runner.

Validates a loaded config dict against the required fields defined in
docs/specs/qstrata_experiment_config_schema.md.

No external validation libraries are used. Field presence is checked
using dot-notation path traversal.

Required fields (as of Q31A formalization):
    experiment.phase        — lifecycle phase identifier
    dataset.name            — dataset identifier
    model.architecture      — architecture family
    reproducibility.seed    — integer random seed
    command.executable      — subprocess executable (e.g. "python3")
    command.args            — list of argument strings for the subprocess

The command block (command.executable + command.args) was added in Q31 to
decouple the runner from model logic, and formalized as a first-class required
field in Q31A. NAS-generated configs (Q32+) produce valid command blocks and
are executed by the runner identically to hand-crafted configs.
"""

from __future__ import annotations

# Minimum required fields for any experiment config.
# Uses dot-notation paths relative to the config dict root.
# command.executable and command.args are first-class required fields (Q31A).
REQUIRED_FIELDS: list[str] = [
    "experiment.phase",
    "dataset.name",
    "model.architecture",
    "reproducibility.seed",
    "command.executable",
    "command.args",
]


def _get_nested(config: dict, dotpath: str) -> tuple[bool, object]:
    """
    Traverse a dot-separated path into a nested dict.

    Returns (found: bool, value: object).
    Returns (False, None) if any key in the path is missing.
    """
    parts = dotpath.split(".")
    node = config
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def validate_config(config: dict) -> None:
    """
    Validate that all required fields are present in the config dict.

    Raises:
        ValueError: listing all missing required fields.
        TypeError: if config is not a dict.
    """
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dict, got {type(config).__name__}")

    missing: list[str] = []
    for field in REQUIRED_FIELDS:
        found, _ = _get_nested(config, field)
        if not found:
            missing.append(field)

    if missing:
        raise ValueError(
            f"Config validation failed. Missing required fields:\n"
            + "\n".join(f"  - {f}" for f in missing)
        )
