"""Launch ML pipelines via SQS for testing.

Usage:
    python launch_pipelines.py --dt                    # Launch 5 random pipelines with DT mode
    python launch_pipelines.py --dt -n 10              # Launch 10 random pipelines
    python launch_pipelines.py --dt --all              # Launch ALL pipelines
    python launch_pipelines.py --dt caco2              # Launch pipelines matching 'caco2'
    python launch_pipelines.py --dt caco2 ppb          # Launch pipelines matching 'caco2' or 'ppb'
    python launch_pipelines.py --promote               # Launch 5 random pipelines with promote mode
    python launch_pipelines.py --promote caco2         # Promote pipelines matching 'caco2'
    python launch_pipelines.py --test-promote --all    # Test-promote ALL pipelines
"""

import argparse
import random
import subprocess
from pathlib import Path

# Relative path to ml_pipelines directory
ML_PIPELINES_DIR = Path(__file__).parent.parent / "ml_pipelines"


def get_all_pipelines() -> list[Path]:
    """Get all ML pipeline scripts from the ml_pipelines directory."""
    return list(ML_PIPELINES_DIR.rglob("*.py"))


def filter_pipelines_by_patterns(pipelines: list[Path], patterns: list[str]) -> list[Path]:
    """Filter pipelines by substring patterns matching the basename."""
    if not patterns:
        return pipelines

    matched = []
    for pipeline in pipelines:
        basename = pipeline.stem.lower()
        if any(pattern.lower() in basename for pattern in patterns):
            matched.append(pipeline)
    return matched


def main():
    parser = argparse.ArgumentParser(description="Launch ML pipelines via SQS for testing")
    parser.add_argument(
        "patterns",
        nargs="*",
        help="Substring patterns to filter pipelines by basename (e.g., 'caco2' 'ppb')",
    )
    parser.add_argument(
        "-n",
        "--num-pipelines",
        type=int,
        default=5,
        help="Number of pipelines to launch (default: 5, ignored if --all or patterns specified)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Launch ALL pipelines (ignores -n)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Create realtime endpoints (default is serverless)",
    )

    # Mode flags (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dt",
        action="store_true",
        help="Launch with DT=True (dynamic training mode)",
    )
    mode_group.add_argument(
        "--promote",
        action="store_true",
        help="Launch with PROMOTE=True (promotion mode)",
    )
    mode_group.add_argument(
        "--test-promote",
        action="store_true",
        help="Launch with TEST_PROMOTE=True (test promotion mode)",
    )

    args = parser.parse_args()

    # Get all pipelines
    all_pipelines = get_all_pipelines()
    if not all_pipelines:
        print(f"No pipeline scripts found in {ML_PIPELINES_DIR}")
        exit(1)

    # Determine which pipelines to run
    if args.patterns:
        # Filter by patterns
        selected_pipelines = filter_pipelines_by_patterns(all_pipelines, args.patterns)
        if not selected_pipelines:
            print(f"No pipelines matching patterns: {args.patterns}")
            exit(1)
        selection_mode = f"matching {args.patterns}"
    elif args.all:
        # Run all pipelines
        selected_pipelines = all_pipelines
        selection_mode = "ALL"
    else:
        # Random selection
        num_to_select = min(args.num_pipelines, len(all_pipelines))
        selected_pipelines = random.sample(all_pipelines, num_to_select)
        selection_mode = f"RANDOM ({args.num_pipelines} requested)"

    # Sort for consistent ordering
    selected_pipelines = sorted(selected_pipelines)

    # Determine mode for display and CLI flag
    if args.dt:
        mode_name = "DT (Dynamic Training)"
        mode_flag = "--dt"
    elif args.promote:
        mode_name = "PROMOTE"
        mode_flag = "--promote"
    else:
        mode_name = "TEST_PROMOTE"
        mode_flag = "--test-promote"

    print(f"\n{'=' * 60}")
    print(f"LAUNCHING {len(selected_pipelines)} PIPELINES")
    print(f"{'=' * 60}")
    print(f"Source: {ML_PIPELINES_DIR}")
    print(f"Selection: {selection_mode}")
    print(f"Mode: {mode_name}")
    print(f"Endpoint: {'Realtime' if args.realtime else 'Serverless'}")
    print("\nSelected pipelines:")
    for i, pipeline in enumerate(selected_pipelines, 1):
        print(f"   {i}. {pipeline.relative_to(ML_PIPELINES_DIR)}")
    print()

    # Launch each pipeline using the CLI
    for i, pipeline in enumerate(selected_pipelines, 1):
        print(f"\n{'─' * 60}")
        print(f"Launching pipeline {i}/{len(selected_pipelines)}: {pipeline.name}")
        print(f"{'─' * 60}")

        # Build the command
        cmd = ["ml_pipeline_sqs", str(pipeline), mode_flag]
        if args.realtime:
            cmd.append("--realtime")

        print(f"Running: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Failed to launch {pipeline.name} (exit code: {result.returncode})")

    print(f"\n{'=' * 60}")
    print(f"FINISHED LAUNCHING {len(selected_pipelines)} PIPELINES")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
