from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from data_generation_constrained_target_depth_8 import DEFAULT_BUDGETS, run_experiment


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = PROJECT_DIR / "data_generation_results_constrained_depth_8_new_target"
DEFAULT_TARGET_SEED = 143
DEFAULT_N_RUNS = 5000


def parse_budgets(value: str) -> List[int]:
    budgets = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not budgets:
        raise argparse.ArgumentTypeError("At least one budget is required.")
    return budgets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PWMCTS data for a second constrained depth-8 target circuit."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--budgets", type=parse_budgets, default=DEFAULT_BUDGETS)
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument(
        "--target-seed",
        type=int,
        default=DEFAULT_TARGET_SEED,
        help="Use a seed different from 43 to create a new target circuit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        out_dir=args.out_dir,
        budgets=args.budgets,
        n_runs=args.n_runs,
        target_seed=args.target_seed,
    )


if __name__ == "__main__":
    main()
