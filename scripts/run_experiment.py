"""
scripts/run_experiment.py

CLI entry point for the QStrata local GPU experiment runner (Q31 MVP).

Usage:
    python scripts/run_experiment.py --config <path-to-yaml>

The script is intentionally thin. All execution logic lives in
qcore/experiments/runner.py.

Arguments:
    --config  Path to the experiment YAML config file (required).

Exit code:
    Passes through the subprocess return code from the configured command.
    0 = success (experiment completed), non-zero = failure.
"""

from __future__ import annotations

import argparse
import sys

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QStrata local experiment runner (Q31 MVP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/run_experiment.py \\\n"
            "    --config configs/experiments/q31_smoke_vindr_cv_binary.yaml\n"
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to experiment YAML config file",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        print(f"[CLI] ERROR: Config file is empty or not valid YAML: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Delegate all logic to the runner
    from qcore.experiments.runner import run_experiment
    return_code = run_experiment(config, config_path=args.config)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
