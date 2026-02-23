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

# Maps CLI arg name -> (mode_override, mode_flag for ml_pipeline_sqs)
MODE_MAP = {
    "dt": ("dt", "--dt"),
    "promote": ("promote", "--promote"),
    "test_promote": ("test_promote", "--test-promote"),
    "temporal_split": ("temporal_split", "--temporal-split"),
}


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
    output_to_input = {}
    for config in configs.values():
        if not config:
            continue
        for output in config.get("outputs", []):
            inputs = config.get("inputs", [])
            output_to_input[output] = inputs[0] if inputs else None

    def find_root(output: str, visited: set = None) -> str:
        if visited is None:
            visited = set()
        if output in visited:
            return output
        visited.add(output)
        parent = output_to_input.get(output)
        return output if parent is None else find_root(parent, visited)

    return {output: find_root(output) for output in output_to_input}


def get_group_id(config: dict | None, root_map: dict[str, str]) -> str | None:
    """Get the root group_id for a pipeline based on its config and root_map."""
    if not config:
        return None
    for key in ("inputs", "outputs"):
        items = config.get(key, [])
        if items and items[0] in root_map:
            return root_map[items[0]]
    return None


def _walk_chain(pipeline: Path, pipelines: set, configs: dict, used: set) -> list[Path]:
    """Walk a dependency chain from a root pipeline, returning the ordered chain."""
    chain = [pipeline]
    used.add(pipeline)
    current = pipeline
    while True:
        current_config = configs.get(current)
        if not current_config or not current_config.get("outputs"):
            break
        current_output = current_config["outputs"][0]
        next_pipeline = next(
            (
                p
                for p, c in configs.items()
                if p not in used and p in pipelines and c and c.get("inputs") and current_output in c["inputs"]
            ),
            None,
        )
        if not next_pipeline:
            break
        chain.append(next_pipeline)
        used.add(next_pipeline)
        current = next_pipeline
    return chain


def _sort_by_workbench_batch(
    pipelines: list[Path],
) -> tuple[list[Path], dict[Path, dict], dict[str, str], list[str]]:
    """Sort pipelines by WORKBENCH_BATCH dependency chains (legacy fallback).

    Returns (sorted_list, configs, root_map, display_lines).
    """
    configs = {p: parse_workbench_batch(p) for p in pipelines}
    root_map = build_dependency_graph(configs)
    pipeline_set = set(pipelines)
    used = set()

    # Walk chains from root producers (no inputs)
    chains = []
    for pipeline in sorted(pipelines):
        config = configs.get(pipeline)
        if pipeline in used or (config and config.get("inputs")):
            continue
        chains.append(_walk_chain(pipeline, pipeline_set, configs, used))

    # Add any remaining pipelines not in chains
    for pipeline in sorted(pipelines):
        if pipeline not in used:
            chains.append([pipeline])

    sorted_pipelines = [p for chain in chains for p in chain]
    display_lines = ["   " + " --> ".join(p.stem for p in chain) for chain in chains]

    return sorted_pipelines, configs, root_map, display_lines


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
                for mode in ([mode_override] if mode_override else modes):
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
        sorted_remaining, configs, root_map, display_lines = _sort_by_workbench_batch(remaining)
        dag_lines.extend(display_lines)
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
        groups.setdefault(pipeline.parent, []).append(pipeline)
    return groups


def select_random_groups(pipelines: list[Path], num_groups: int) -> list[Path]:
    """Select pipelines from n random leaf directories."""
    groups = get_pipeline_groups(pipelines)
    if not groups:
        return []
    selected_dirs = random.sample(list(groups), min(num_groups, len(groups)))
    return [p for d in selected_dirs for p in groups[d]]


def filter_pipelines_by_patterns(pipelines: list[Path], patterns: list[str]) -> list[Path]:
    """Filter pipelines by substring patterns matching the basename."""
    if not patterns:
        return pipelines
    patterns_lower = [p.lower() for p in patterns]
    return [p for p in pipelines if any(pat in p.stem.lower() for pat in patterns_lower)]


def main():
    parser = argparse.ArgumentParser(description="Launch ML pipelines via SQS or locally")
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
    parser.add_argument("--all", action="store_true", help="Launch ALL pipelines (ignores -n)")
    parser.add_argument("--realtime", action="store_true", help="Create realtime endpoints (default is serverless)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be launched without actually launching")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run pipelines locally instead of via SQS (uses active Python interpreter)",
    )

    # Mode flags (mutually exclusive) — override yaml default modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dt", action="store_true", help="Override all scripts to DT mode (dynamic training)")
    mode_group.add_argument("--promote", action="store_true", help="Override all scripts to PROMOTE mode")
    mode_group.add_argument("--test-promote", action="store_true", help="Override all scripts to TEST_PROMOTE mode")
    mode_group.add_argument(
        "--temporal-split", action="store_true", help="Override all scripts to temporal split evaluation mode"
    )

    args = parser.parse_args()

    # Get all pipelines and DAG definitions from subdirectories
    all_pipelines, all_dags = get_all_pipelines()
    if not all_pipelines:
        print(f"No pipeline scripts found in subdirectories of {Path.cwd()}")
        exit(1)

    # Determine which pipelines to run
    if args.patterns:
        selected_pipelines = filter_pipelines_by_patterns(all_pipelines, args.patterns)
        if not selected_pipelines:
            print(f"No pipelines matching patterns: {args.patterns}")
            exit(1)
        selection_mode = f"matching {args.patterns}"
    elif args.all:
        selected_pipelines = all_pipelines
        selection_mode = "ALL"
    else:
        selected_pipelines = select_random_groups(all_pipelines, args.num_groups)
        if not selected_pipelines:
            print("No pipeline groups found")
            exit(1)
        group_names = [d.name for d in get_pipeline_groups(selected_pipelines)]
        selection_mode = f"RANDOM {args.num_groups} group(s): {group_names}"

    # Determine mode override from CLI (None means use yaml defaults)
    mode_override, mode_flag = None, None
    for arg_name, (override, flag) in MODE_MAP.items():
        if getattr(args, arg_name, False):
            mode_override, mode_flag = override, flag
            break

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

        # Build per-script PIPELINE_META with resolved names
        pipeline_meta = build_pipeline_meta(script, mode, not args.realtime)

        if args.local:
            env = os.environ.copy()
            env["PIPELINE_META"] = pipeline_meta
            cmd = [sys.executable, str(script)]
            print(f"with ENV: PIPELINE_META='{pipeline_meta}'")
            print(f"{'─' * 60}\n")
            result = subprocess.run(cmd, env=env)
        else:
            run_mode_flag = mode_flag or f"--{mode.replace('_', '-')}"
            cmd = ["ml_pipeline_sqs", str(script), run_mode_flag]
            if args.realtime:
                cmd.append("--realtime")
            cmd.extend(["--pipeline-meta", pipeline_meta])
            group_id = group_id_map.get((script, mode))
            if group_id:
                cmd.extend(["--group-id", group_id])
            print(f"{'─' * 60}")
            print(f"Running: {' '.join(cmd)}\n")
            result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"Failed to launch {script.name} (exit code: {result.returncode})")

    print(f"\n{'=' * 60}")
    print(f"FINISHED LAUNCHING {len(sorted_runs)} PIPELINE RUNS")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
