"""Launch ML pipelines via SQS or locally.

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
    ml_pipeline_launcher --local --dt ppb_human  # Run pipelines locally (uses active Python interpreter)
"""

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

VERSION_RE = re.compile(r"^(.+?)_v?(\d+)$")
FRAMEWORK_RE = re.compile(r"-(xgb|pytorch|chemprop|chemeleon)$")


def parse_script_name(script_path: Path) -> tuple[str, str]:
    """Parse a pipeline script filename into (basename, version).

    Filenames must end with _{number} or _v{number} (e.g., my_script_1.py, my_script_v2.py).

    Args:
        script_path (Path): Path to the pipeline script

    Returns:
        tuple[str, str]: (basename, version) — e.g., ("ppb_human_free_reg_xgb", "1")
    """
    match = VERSION_RE.match(script_path.stem)
    if not match:
        raise RuntimeError(
            f"Pipeline script filename must end with _{{number}}.py or _v{{number}}.py: {script_path.name}"
        )
    return match.group(1), match.group(2)


def build_pipeline_meta(script_path: Path, mode: str, serverless: bool) -> str:
    """Build PIPELINE_META JSON for a pipeline script.

    Derives model_name and endpoint_name from the script filename and mode.
    For promoted endpoints, the framework suffix (xgb, pytorch, etc.) is stripped.
    """
    from datetime import datetime

    basename, version = parse_script_name(script_path)
    basename_hyphen = basename.replace("_", "-")  # e.g., "ppb-human-free-reg-xgb"
    endpoint_base = FRAMEWORK_RE.sub("", basename_hyphen)  # e.g., "ppb-human-free-reg"
    today = datetime.now().strftime("%y%m%d")

    if mode in ("dt", "temporal_split"):
        model_name = f"{basename_hyphen}-{version}-dt"
        endpoint_name = f"{basename_hyphen}-{version}-dt"
    elif mode == "promote":
        model_name = f"{basename_hyphen}-{version}-{today}"
        endpoint_name = f"{endpoint_base}-{version}"
    elif mode == "test_promote":
        model_name = f"{basename_hyphen}-{version}-{today}"
        endpoint_name = f"{endpoint_base}-{version}-test"
    else:
        raise RuntimeError(f"Unknown mode: {mode}")

    return json.dumps(
        {"mode": mode, "model_name": model_name, "endpoint_name": endpoint_name, "serverless": serverless}
    )


def load_pipelines_yaml(directory: Path) -> dict[str, list[dict[Path, list[str]]]] | None:
    """Load pipelines.yaml from a directory.

    The yaml uses a stage-based DAG format where each list item is a stage
    (dict of script: [modes]). Scripts within a stage can run in parallel;
    stages run sequentially.

    Args:
        directory (Path): Directory to check for pipelines.yaml

    Returns:
        dict | None: {dag_name: [stages]} where each stage is {script_path: [modes]},
            or None if no yaml found
    """
    yaml_path = directory / "pipelines.yaml"
    if not yaml_path.exists():
        return None
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    dags = {}
    for dag_name, stages in config.get("dags", {}).items():
        dag_stages = []
        for stage in stages:
            stage_dict = {directory / script: modes for script, modes in stage.items()}
            dag_stages.append(stage_dict)
        dags[dag_name] = dag_stages
    return dags


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
    pipelines: list[Path],
    all_dags: dict[str, list[dict[Path, list[str]]]],
    mode_override: str | None = None,
) -> tuple[list[tuple[Path, str]], dict[tuple[Path, str], str | None], list[str]]:
    """Sort pipelines by DAG stages with per-script modes.

    Uses yaml DAGs when available, falls back to WORKBENCH_BATCH parsing for
    pipelines not covered by any yaml DAG.

    Args:
        pipelines (list[Path]): Pipelines to sort
        all_dags (dict): DAG definitions from pipelines.yaml files
            {dag_name: [stages]} where each stage is {script_path: [modes]}
        mode_override (str | None): If set, overrides all yaml modes (from CLI flags)

    Returns:
        tuple: (sorted_runs, group_id_map, dag_lines)
            - sorted_runs: List of (script_path, mode) tuples in execution order
            - group_id_map: {(script_path, mode): sqs_message_group_id}
            - dag_lines: Formatted display lines
    """
    pipeline_set = set(pipelines)
    sorted_runs = []
    group_id_map = {}
    dag_lines = []
    used = set()

    # First: process pipelines that are in yaml DAGs
    for dag_name, stages in all_dags.items():
        dag_stage_lines = []
        dag_has_runs = False
        for stage in stages:
            stage_parts = []
            for script, modes in stage.items():
                if script not in pipeline_set:
                    continue
                used.add(script)
                dag_has_runs = True
                run_modes = [mode_override] if mode_override else modes
                for mode in run_modes:
                    run = (script, mode)
                    sorted_runs.append(run)
                    group_id_map[run] = dag_name
                    stage_parts.append(f"{script.stem}:{mode}")
            if stage_parts:
                dag_stage_lines.append(" | ".join(stage_parts))
        if dag_has_runs:
            dag_lines.append("   " + " --> ".join(dag_stage_lines))

    # Second: process remaining pipelines via WORKBENCH_BATCH fallback
    remaining = [p for p in pipelines if p not in used]
    if remaining:
        fallback_mode = mode_override or "dt"
        sorted_remaining, configs, root_map = _sort_by_workbench_batch(remaining)
        for line in _format_workbench_batch_chains(sorted_remaining, configs):
            dag_lines.append(line)
        for pipeline in sorted_remaining:
            run = (pipeline, fallback_mode)
            sorted_runs.append(run)
            group_id_map[run] = get_group_id(configs.get(pipeline), root_map)

    return sorted_runs, group_id_map, dag_lines


def get_all_pipelines() -> tuple[list[Path], dict[str, list[dict[Path, list[str]]]]]:
    """Get all ML pipeline scripts from subdirectories of current working directory.

    For directories with pipelines.yaml, only listed scripts are included.
    For directories without, falls back to discovering all .py files.

    Returns:
        tuple: (pipelines, all_dags)
            - pipelines: List of unique pipeline script paths
            - all_dags: {dag_name: [stages]} from yaml files
    """
    cwd = Path.cwd()
    pipelines = []
    all_dags = {}
    yaml_managed_dirs = set()
    seen_scripts = set()

    # First pass: find all pipelines.yaml files recursively
    for yaml_path in cwd.rglob("pipelines.yaml"):
        directory = yaml_path.parent
        dag_defs = load_pipelines_yaml(directory)
        if dag_defs:
            yaml_managed_dirs.add(directory)
            all_dags.update(dag_defs)
            for stages in dag_defs.values():
                for stage in stages:
                    for script in stage.keys():
                        if script not in seen_scripts:
                            pipelines.append(script)
                            seen_scripts.add(script)

    # Second pass: find .py files in directories NOT managed by yaml
    for subdir in cwd.iterdir():
        if subdir.is_dir():
            for py_file in subdir.rglob("*.py"):
                if py_file.parent not in yaml_managed_dirs:
                    pipelines.append(py_file)

    return pipelines, all_dags


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
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run pipelines locally instead of via SQS (uses active Python interpreter)",
    )

    # Mode flags (mutually exclusive) — override yaml default modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dt",
        action="store_true",
        help="Override all scripts to DT mode (dynamic training)",
    )
    mode_group.add_argument(
        "--promote",
        action="store_true",
        help="Override all scripts to PROMOTE mode",
    )
    mode_group.add_argument(
        "--test-promote",
        action="store_true",
        help="Override all scripts to TEST_PROMOTE mode",
    )
    mode_group.add_argument(
        "--temporal-split",
        action="store_true",
        help="Override all scripts to temporal split evaluation mode",
    )

    args = parser.parse_args()

    # Get all pipelines and DAG definitions from subdirectories
    all_pipelines, all_dags = get_all_pipelines()
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

    # Determine mode override from CLI (None means use yaml defaults)
    mode_override = None
    mode_flag = None
    if args.dt:
        mode_override = "dt"
        mode_flag = "--dt"
    elif args.promote:
        mode_override = "promote"
        mode_flag = "--promote"
    elif args.test_promote:
        mode_override = "test_promote"
        mode_flag = "--test-promote"
    elif args.temporal_split:
        mode_override = "temporal_split"
        mode_flag = "--temporal-split"

    # Sort by DAG stages (with mode override if CLI flag set)
    sorted_runs, group_id_map, dag_lines = sort_pipelines(selected_pipelines, all_dags, mode_override)

    # Local mode only supports a single script
    if args.local and len(selected_pipelines) > 1:
        print(f"\n--local only supports a single script, but {len(selected_pipelines)} matched:")
        for p in selected_pipelines:
            print(f"   {p.name}")
        print("\nNarrow your selection to a single script.")
        exit(1)

    mode_name = mode_override.upper() if mode_override else "YAML defaults"

    print(f"\n{'=' * 60}")
    print(f"{'DRY RUN - ' if args.dry_run else ''}LAUNCHING {len(sorted_runs)} PIPELINE RUNS")
    print(f"{'=' * 60}")
    print(f"Source: {Path.cwd()}")
    print(f"Selection: {selection_mode}")
    print(f"Mode: {mode_name}")
    print(f"Execution: {'Local' if args.local else 'SQS → Batch'}")
    print(f"Endpoint: {'Realtime' if args.realtime else 'Serverless'}")
    print("\nPipeline DAGs:")
    for line in dag_lines:
        print(line)
    print()

    # Dry run - just show what would be launched
    if args.dry_run:
        print("Dry run complete. No pipelines were launched.\n")
        return

    # Countdown before launching (skip for local runs)
    if not args.local:
        print("Launching in ", end="", flush=True)
        for i in range(10, 0, -1):
            print(f"{i}...", end="", flush=True)
            time.sleep(1)
        print(" GO!\n")

    # Launch each pipeline run
    for i, (script, mode) in enumerate(sorted_runs, 1):
        print(f"\n{'─' * 60}")
        print(f"{'Running' if args.local else 'Launching'} run {i}/{len(sorted_runs)}: {script.name} ({mode})")
        print(f"{'─' * 60}")

        # Build per-script PIPELINE_META with resolved names
        pipeline_meta = build_pipeline_meta(script, mode, not args.realtime)

        if args.local:
            # Run locally with PIPELINE_META set in the environment
            env = os.environ.copy()
            env["PIPELINE_META"] = pipeline_meta
            cmd = [sys.executable, str(script)]
            print(f"with ENV: PIPELINE_META='{pipeline_meta}'")
            print(f"{'─' * 60}\n")
            result = subprocess.run(cmd, env=env)
        else:
            # Launch via SQS → Batch
            run_mode_flag = mode_flag or f"--{mode.replace('_', '-')}"
            cmd = ["ml_pipeline_sqs", str(script), run_mode_flag]
            if args.realtime:
                cmd.append("--realtime")
            cmd.extend(["--pipeline-meta", pipeline_meta])

            # Pass group_id for dependency chain ordering (from yaml DAG name or WORKBENCH_BATCH root)
            group_id = group_id_map.get((script, mode))
            if group_id:
                cmd.extend(["--group-id", group_id])

            print(f"Running: {' '.join(cmd)}\n")
            result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"Failed to launch {script.name} (exit code: {result.returncode})")

    print(f"\n{'=' * 60}")
    print(f"FINISHED LAUNCHING {len(sorted_runs)} PIPELINE RUNS")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
