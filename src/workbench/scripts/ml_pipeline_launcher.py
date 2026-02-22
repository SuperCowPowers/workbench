"""Launch ML pipelines via SQS for testing.

Run this from a directory containing pipeline subdirectories (e.g., ml_pipelines/).

Usage:
    ml_pipeline_launcher --dt                    # Launch 1 random pipeline group (all scripts in a directory)
    ml_pipeline_launcher --dt -n 3               # Launch 3 random pipeline groups
    ml_pipeline_launcher --dt --all              # Launch ALL pipelines
    ml_pipeline_launcher --dt caco2              # Launch pipelines matching 'caco2'
    ml_pipeline_launcher --dt caco2 ppb          # Launch pipelines matching 'caco2' or 'ppb'
    ml_pipeline_launcher --promote --all         # Promote ALL pipelines
    ml_pipeline_launcher --test-promote --all    # Test-promote ALL pipelines
    ml_pipeline_launcher --temporal-split --all  # Temporal split ALL pipelines
    ml_pipeline_launcher --dt --dry-run          # Show what would be launched without launching
"""

import argparse
import ast
import json
import random
import re
import subprocess
import time
from pathlib import Path

import yaml


FRAMEWORKS_RE = re.compile(r"-(xgb|pytorch|chemprop|chemeleon)-\d+$")


def build_pipeline_meta(script_path: Path, mode: str, serverless: bool) -> str:
    """Build PIPELINE_META JSON for a pipeline script.

    Derives model_name and endpoint_name from the script filename and mode.
    For promoted endpoints, the framework suffix (xgb, pytorch, etc.) is stripped.
    """
    from datetime import datetime

    model_name_base = script_path.stem.replace("_", "-")
    endpoint_name_base = FRAMEWORKS_RE.sub("", model_name_base)
    today = datetime.now().strftime("%y%m%d")

    if mode in ("dt", "temporal_split"):
        model_name = f"{model_name_base}-dt"
        endpoint_name = f"{model_name_base}-dt"
    elif mode == "promote":
        model_name = f"{model_name_base}-{today}"
        endpoint_name = f"{endpoint_name_base}-1"
    elif mode == "test_promote":
        model_name = f"{model_name_base}-{today}"
        endpoint_name = f"{endpoint_name_base}-1-test"
    else:  # dev
        model_name = f"{model_name_base}-{today}-test"
        endpoint_name = f"{endpoint_name_base}-{today}-test"

    return json.dumps({"mode": mode, "model_name": model_name, "endpoint_name": endpoint_name, "serverless": serverless})


def load_pipelines_yaml(directory: Path) -> dict[str, list[Path]] | None:
    """Load pipelines.yaml from a directory.

    Args:
        directory (Path): Directory to check for pipelines.yaml

    Returns:
        dict[str, list[Path]] | None: {chain_name: [script_paths]} or None if no yaml found
    """
    yaml_path = directory / "pipelines.yaml"
    if not yaml_path.exists():
        return None
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    chains = {}
    for chain_name, scripts in config.get("chains", {}).items():
        chains[chain_name] = [directory / s for s in scripts]
    return chains


def parse_workbench_batch(script_path: Path) -> dict | None:
    """Parse WORKBENCH_BATCH config from a script file."""
    content = script_path.read_text()
    match = re.search(r"WORKBENCH_BATCH\s*=\s*(\{[^}]+\})", content, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            return None
    return None


def build_dependency_graph(configs: dict[Path, dict]) -> dict[str, str]:
    """Build a mapping from each output to its root producer.

    For a chain like A -> B -> C (where B depends on A, C depends on B),
    this returns {A: A, B: A, C: A} so all are in the same message group.
    """
    # Build output -> input mapping (what does each output depend on?)
    output_to_input = {}
    for config in configs.values():
        if not config:
            continue
        outputs = config.get("outputs", [])
        inputs = config.get("inputs", [])
        for output in outputs:
            output_to_input[output] = inputs[0] if inputs else None

    # Walk chain to find root
    def find_root(output: str, visited: set = None) -> str:
        if visited is None:
            visited = set()
        if output in visited:
            return output
        visited.add(output)
        parent = output_to_input.get(output)
        if parent is None:
            return output
        return find_root(parent, visited)

    return {output: find_root(output) for output in output_to_input}


def get_group_id(config: dict | None, root_map: dict[str, str]) -> str | None:
    """Get the root group_id for a pipeline based on its config and root_map."""
    if not config:
        return None
    outputs = config.get("outputs", [])
    inputs = config.get("inputs", [])
    # Check inputs first (this script depends on something)
    if inputs and inputs[0] in root_map:
        return root_map[inputs[0]]
    # Check outputs (this script produces something)
    if outputs and outputs[0] in root_map:
        return root_map[outputs[0]]
    return None


def _sort_by_workbench_batch(pipelines: list[Path]) -> tuple[list[Path], dict[Path, dict], dict[str, str]]:
    """Sort pipelines by WORKBENCH_BATCH dependency chains (legacy fallback).

    Returns (sorted_list, configs, root_map).
    """
    # Parse all configs
    configs = {}
    for pipeline in pipelines:
        configs[pipeline] = parse_workbench_batch(pipeline)

    # Build root map for group_id resolution
    root_map = build_dependency_graph(configs)

    # Build output -> pipeline mapping
    output_to_pipeline = {}
    for pipeline, config in configs.items():
        if config and config.get("outputs"):
            for output in config["outputs"]:
                output_to_pipeline[output] = pipeline

    # Build chains by walking from root producers
    sorted_pipelines = []
    used = set()

    for pipeline in sorted(pipelines):
        config = configs.get(pipeline)

        # Skip if already used or has inputs (not a root)
        if pipeline in used:
            continue
        if config and config.get("inputs"):
            continue

        # Walk the chain from this root
        chain = [pipeline]
        used.add(pipeline)

        current = pipeline
        while True:
            current_config = configs.get(current)
            if not current_config or not current_config.get("outputs"):
                break

            current_output = current_config["outputs"][0]
            # Find pipeline that consumes this output
            next_pipeline = None
            for p, c in configs.items():
                if p in used or p not in pipelines:
                    continue
                if c and c.get("inputs") and current_output in c["inputs"]:
                    next_pipeline = p
                    break

            if next_pipeline:
                chain.append(next_pipeline)
                used.add(next_pipeline)
                current = next_pipeline
            else:
                break

        sorted_pipelines.extend(chain)

    # Add any remaining pipelines not in chains
    for pipeline in sorted(pipelines):
        if pipeline not in used:
            sorted_pipelines.append(pipeline)

    return sorted_pipelines, configs, root_map


def _format_workbench_batch_chains(pipelines: list[Path], configs: dict[Path, dict]) -> list[str]:
    """Format pipelines as dependency chains for display (legacy fallback)."""
    # Build output -> pipeline mapping
    output_to_pipeline = {}
    for pipeline, config in configs.items():
        if config and config.get("outputs"):
            for output in config["outputs"]:
                output_to_pipeline[output] = pipeline

    # Build chains by walking from root producers
    chains = []
    used = set()

    for pipeline in pipelines:
        config = configs.get(pipeline)

        # Skip if already part of a chain or has inputs (not a root)
        if pipeline in used:
            continue
        if config and config.get("inputs"):
            continue

        # Start a new chain from this root producer (or standalone)
        chain = [pipeline]
        used.add(pipeline)

        # Walk the chain: find who consumes our output
        current = pipeline
        while True:
            current_config = configs.get(current)
            if not current_config or not current_config.get("outputs"):
                break

            current_output = current_config["outputs"][0]
            # Find a pipeline that takes this output as input
            next_pipeline = None
            for p, c in configs.items():
                if p in used or p not in pipelines:
                    continue
                if c and c.get("inputs") and current_output in c["inputs"]:
                    next_pipeline = p
                    break

            if next_pipeline:
                chain.append(next_pipeline)
                used.add(next_pipeline)
                current = next_pipeline
            else:
                break

        chains.append(chain)

    # Add any remaining pipelines not in chains (shouldn't happen but just in case)
    for pipeline in pipelines:
        if pipeline not in used:
            chains.append([pipeline])

    # Format chains as strings
    lines = []
    for chain in chains:
        names = [p.stem for p in chain]
        lines.append("   " + " --> ".join(names))

    return lines


def sort_pipelines(
    pipelines: list[Path], all_chains: dict[str, list[Path]]
) -> tuple[list[Path], dict[Path, str | None], list[str]]:
    """Sort pipelines by dependency chains.

    Uses yaml chains when available, falls back to WORKBENCH_BATCH parsing for
    pipelines not covered by any yaml chain.

    Args:
        pipelines (list[Path]): Pipelines to sort
        all_chains (dict[str, list[Path]]): Chain definitions from pipelines.yaml files

    Returns:
        tuple: (sorted_pipelines, group_id_map, chain_lines)
            - sorted_pipelines: Pipelines ordered by dependency chains
            - group_id_map: {pipeline_path: sqs_message_group_id}
            - chain_lines: Formatted display lines (e.g., "   xgb --> pytorch --> chemprop")
    """
    pipeline_set = set(pipelines)
    sorted_pipelines = []
    group_id_map = {}
    chain_lines = []
    used = set()

    # First: process pipelines that are in yaml chains
    for chain_name, chain_scripts in all_chains.items():
        # Only include scripts that are in the selected pipelines
        selected_chain = [s for s in chain_scripts if s in pipeline_set]
        if selected_chain:
            chain_lines.append("   " + " --> ".join(s.stem for s in selected_chain))
            for script in selected_chain:
                sorted_pipelines.append(script)
                group_id_map[script] = chain_name
                used.add(script)

    # Second: process remaining pipelines via WORKBENCH_BATCH fallback
    remaining = [p for p in pipelines if p not in used]
    if remaining:
        sorted_remaining, configs, root_map = _sort_by_workbench_batch(remaining)
        for line in _format_workbench_batch_chains(sorted_remaining, configs):
            chain_lines.append(line)
        for pipeline in sorted_remaining:
            sorted_pipelines.append(pipeline)
            group_id_map[pipeline] = get_group_id(configs.get(pipeline), root_map)

    return sorted_pipelines, group_id_map, chain_lines


def get_all_pipelines() -> tuple[list[Path], dict[str, list[Path]]]:
    """Get all ML pipeline scripts from subdirectories of current working directory.

    For directories with pipelines.yaml, only listed scripts are included.
    For directories without, falls back to discovering all .py files.

    Returns:
        tuple: (pipelines, all_chains)
            - pipelines: List of all pipeline script paths
            - all_chains: {chain_name: [script_paths]} from yaml files
    """
    cwd = Path.cwd()
    pipelines = []
    all_chains = {}
    yaml_managed_dirs = set()

    # First pass: find all pipelines.yaml files recursively
    for yaml_path in cwd.rglob("pipelines.yaml"):
        directory = yaml_path.parent
        yaml_chains = load_pipelines_yaml(directory)
        if yaml_chains:
            yaml_managed_dirs.add(directory)
            all_chains.update(yaml_chains)
            for chain_scripts in yaml_chains.values():
                pipelines.extend(chain_scripts)

    # Second pass: find .py files in directories NOT managed by yaml
    for subdir in cwd.iterdir():
        if subdir.is_dir():
            for py_file in subdir.rglob("*.py"):
                if py_file.parent not in yaml_managed_dirs:
                    pipelines.append(py_file)

    return pipelines, all_chains


def get_pipeline_groups(pipelines: list[Path]) -> dict[Path, list[Path]]:
    """Group pipelines by their parent directory (leaf directories)."""
    groups = {}
    for pipeline in pipelines:
        parent = pipeline.parent
        groups.setdefault(parent, []).append(pipeline)
    return groups


def select_random_groups(pipelines: list[Path], num_groups: int) -> list[Path]:
    """Select pipelines from n random leaf directories."""
    groups = get_pipeline_groups(pipelines)
    if not groups:
        return []

    # Select up to num_groups random directories
    dirs = list(groups.keys())
    selected_dirs = random.sample(dirs, min(num_groups, len(dirs)))

    # Return all pipelines from those directories
    selected = []
    for d in selected_dirs:
        selected.extend(groups[d])
    return selected


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
        "--num-groups",
        type=int,
        default=1,
        help="Number of random pipeline groups to launch (default: 1, ignored if --all or patterns specified)",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be launched without actually launching",
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
    mode_group.add_argument(
        "--temporal-split",
        action="store_true",
        help="Launch with temporal split evaluation mode",
    )

    args = parser.parse_args()

    # Get all pipelines and chain definitions from subdirectories
    all_pipelines, all_chains = get_all_pipelines()
    if not all_pipelines:
        print(f"No pipeline scripts found in subdirectories of {Path.cwd()}")
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
        # Random group selection
        selected_pipelines = select_random_groups(all_pipelines, args.num_groups)
        if not selected_pipelines:
            print("No pipeline groups found")
            exit(1)
        # Get the directory names for display
        groups = get_pipeline_groups(selected_pipelines)
        group_names = [d.name for d in groups.keys()]
        selection_mode = f"RANDOM {args.num_groups} group(s): {group_names}"

    # Sort by dependencies (producers before consumers)
    selected_pipelines, group_id_map, chain_lines = sort_pipelines(selected_pipelines, all_chains)

    # Determine mode for display and CLI flag
    if args.dt:
        mode_name = "DT (Dynamic Training)"
        mode_flag = "--dt"
    elif args.promote:
        mode_name = "PROMOTE"
        mode_flag = "--promote"
    elif args.temporal_split:
        mode_name = "TEMPORAL_SPLIT"
        mode_flag = "--temporal-split"
    else:
        mode_name = "TEST_PROMOTE"
        mode_flag = "--test-promote"

    # Determine pipeline mode
    mode_map = {"dt": args.dt, "promote": args.promote, "test_promote": args.test_promote, "temporal_split": args.temporal_split}
    pipeline_mode = next((mode for mode, flag in mode_map.items() if flag), "dev")

    print(f"\n{'=' * 60}")
    print(f"{'DRY RUN - ' if args.dry_run else ''}LAUNCHING {len(selected_pipelines)} PIPELINES")
    print(f"{'=' * 60}")
    print(f"Source: {Path.cwd()}")
    print(f"Selection: {selection_mode}")
    print(f"Mode: {mode_name}")
    print(f"Endpoint: {'Realtime' if args.realtime else 'Serverless'}")
    print("\nPipeline Chains:")
    for line in chain_lines:
        print(line)
    print()

    # Dry run - just show what would be launched
    if args.dry_run:
        print("Dry run complete. No pipelines were launched.\n")
        return

    # Countdown before launching
    print("Launching in ", end="", flush=True)
    for i in range(10, 0, -1):
        print(f"{i}...", end="", flush=True)
        time.sleep(1)
    print(" GO!\n")

    # Launch each pipeline using the CLI
    for i, pipeline in enumerate(selected_pipelines, 1):
        print(f"\n{'─' * 60}")
        print(f"Launching pipeline {i}/{len(selected_pipelines)}: {pipeline.name}")
        print(f"{'─' * 60}")

        # Build per-script PIPELINE_META with resolved names
        pipeline_meta = build_pipeline_meta(pipeline, pipeline_mode, not args.realtime)

        # Build the command
        cmd = ["ml_pipeline_sqs", str(pipeline), mode_flag]
        if args.realtime:
            cmd.append("--realtime")
        cmd.extend(["--pipeline-meta", pipeline_meta])

        # Pass group_id for dependency chain ordering (from yaml chain name or WORKBENCH_BATCH root)
        group_id = group_id_map.get(pipeline)
        if group_id:
            cmd.extend(["--group-id", group_id])

        print(f"Running: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Failed to launch {pipeline.name} (exit code: {result.returncode})")

    print(f"\n{'=' * 60}")
    print(f"FINISHED LAUNCHING {len(selected_pipelines)} PIPELINES")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
