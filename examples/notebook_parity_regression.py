"""
Notebook-parity regression benchmark runner.

Runs representative V18-like and V20.2b-like Split-MNIST profiles against the
current FabricPC port and optionally compares the results against a checked-in
baseline file.

Usage:
    python examples/notebook_parity_regression.py
    python examples/notebook_parity_regression.py --profile v18_like
    python examples/notebook_parity_regression.py --update-baseline
"""

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cpu")

import argparse
from pathlib import Path

from fabricpc.continual.parity import (
    PROFILES,
    compare_against_baseline,
    load_parity_baselines,
    run_parity_profile,
    save_parity_baselines,
)


def main():
    parser = argparse.ArgumentParser(description="Notebook-parity regression runner")
    parser.add_argument(
        "--profile",
        type=str,
        choices=sorted(PROFILES.keys()),
        default=None,
        help="Run a single parity profile instead of all profiles",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="MNIST data root",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="benchmarks/notebook_parity_baselines.json",
        help="Path to the baseline JSON file",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite baseline file with the metrics from this run",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_path)
    profile_names = [args.profile] if args.profile else list(PROFILES.keys())

    metrics_by_profile = {}
    for profile_name in profile_names:
        profile = PROFILES[profile_name]
        print(f"\n=== {profile.name} ===")
        print(profile.description)
        metrics = run_parity_profile(
            profile_name,
            seed=args.seed,
            data_root=args.data_root,
        )
        metrics_by_profile[profile_name] = metrics
        print(metrics)

    if args.update_baseline:
        save_parity_baselines(metrics_by_profile, baseline_path)
        print(f"\nUpdated baseline: {baseline_path}")
        return

    baselines = load_parity_baselines(baseline_path)
    all_passed = True
    for profile_name, metrics in metrics_by_profile.items():
        comparison = compare_against_baseline(metrics, baselines[profile_name])
        print(
            f"\nComparison for {profile_name}: {'PASS' if comparison.passed else 'FAIL'}"
        )
        for check in comparison.checks:
            print(
                f"  {check.metric}: observed={check.observed:.4f} "
                f"expected={check.expected:.4f} diff={check.abs_diff:.4f} "
                f"tol={check.abs_tol:.4f} {'OK' if check.passed else 'OUT'}"
            )
        all_passed = all_passed and comparison.passed

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
