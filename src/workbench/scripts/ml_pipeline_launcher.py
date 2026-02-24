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
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

VERSION_RE = re.compile(r"^(.+?)_v?(\d+)$")
FRAMEWORK_RE = re.compile(r"-(xgb|pytorch|chemprop|chemeleon)$")

# Maps CLI arg name -> (mode_override, mode_flag for ml_pipeline_sqs)
MODE_MAP = {
    "dt": ("dt", "--dt"),
    "promote": ("promote", "--promote"),
    "test_promote": ("test_promote", "--test-promote"),
    "temporal_split": ("ts", "--temporal-split"),
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

    if mode in ("dt", "ts"):
        model_name = f"{basename_hyphen}-{version}-{mode}"
        endpoint_name = f"{basename_hyphen}-{version}-{mode}"
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


def load_pipelines_config(directory: Path) -> dict[str, list[dict[Path, list[str]]]] | None:
    """Load pipelines.json from a directory.

    The JSON uses a stage-based DAG format where each list item is a stage
    (dict of script: [modes]). Scripts within a stage can run in parallel;
    stages run sequentially.

    Args:
        directory (Path): Directory to check for pipelines.json

    Returns:
        dict | None: {dag_name: [stages]} where each stage is {script_path: [modes]},
            or None if no JSON config found
    """
    json_path = directory / "pipelines.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        config = json.load(f)
    dags = {}
    for dag_name, stages in config.get("dags", {}).items():
        dag_stages = []
        for stage in stages:
            stage_dict = {directory / script: modes for script, modes in stage.items()}
            dag_stages.append(stage_dict)
        dags[dag_name] = dag_stages
    return dags


def sort_pipelines(
    pipelines: list[Path],
    all_dags: dict[str, list[dict[Path, list[str]]]],
    mode_override: str | None = None,
) -> tuple[list[tuple[Path, str]], dict[tuple[Path, str], str | None], list[str], dict]:
    """Sort pipelines by DAG stages with per-script modes.

    All pipelines must be defined in a pipelines.json DAG.

    Args:
        pipelines (list[Path]): Pipelines to sort
        all_dags (dict): DAG definitions from pipelines.json files
            {dag_name: [stages]} where each stage is {script_path: [modes]}
        mode_override (str | None): If set, overrides all JSON modes (from CLI flags)

    Returns:
        tuple: (sorted_runs, group_id_map, dag_lines, deps_map)
            - sorted_runs: List of (script_path, mode) tuples in execution order
            - group_id_map: {(script_path, mode): sqs_message_group_id}
            - dag_lines: Formatted display lines
            - deps_map: {(script_path, mode): {"outputs": list, "inputs": list}}
    """
    pipeline_set = set(pipelines)
    sorted_runs = []
    group_id_map = {}
    deps_map = {}
    dag_lines = []

    for dag_name, stages in all_dags.items():
        dag_stage_lines = []
        dag_has_runs = False
        for stage_idx, stage in enumerate(stages):
            stage_parts = []
            outputs = [f"{dag_name}:stage_{stage_idx}"]
            inputs = [f"{dag_name}:stage_{stage_idx - 1}"] if stage_idx > 0 else []
            for script, modes in stage.items():
                if script not in pipeline_set:
                    continue
                dag_has_runs = True
                for mode in ([mode_override] if mode_override else modes):
                    run = (script, mode)
                    sorted_runs.append(run)
                    group_id_map[run] = dag_name
                    deps_map[run] = {"outputs": outputs, "inputs": inputs}
                    stage_parts.append(f"{script.stem}:{mode}")
            if stage_parts:
                dag_stage_lines.append(" | ".join(stage_parts))
        if dag_has_runs:
            dag_lines.append("   " + " --> ".join(dag_stage_lines))

    return sorted_runs, group_id_map, dag_lines, deps_map


def get_all_pipelines() -> tuple[list[Path], dict[str, list[dict[Path, list[str]]]]]:
    """Get all ML pipeline scripts from pipelines.json files in subdirectories.

    Only scripts listed in a pipelines.json are included.

    Returns:
        tuple: (pipelines, all_dags)
            - pipelines: List of unique pipeline script paths
            - all_dags: {dag_name: [stages]} from pipelines.json files
    """
    cwd = Path.cwd()
    pipelines = []
    all_dags = {}
    seen_scripts = set()

    for config_path in cwd.rglob("pipelines.json"):
        directory = config_path.parent
        dag_defs = load_pipelines_config(directory)
        if dag_defs:
            all_dags.update(dag_defs)
            for stages in dag_defs.values():
                for stage in stages:
                    for script in stage.keys():
                        if script not in seen_scripts:
                            pipelines.append(script)
                            seen_scripts.add(script)

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

    # Mode flags (mutually exclusive) — override pipelines.json default modes
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

    # Determine mode override from CLI (None means use pipelines.json defaults)
    mode_override, mode_flag = None, None
    for arg_name, (override, flag) in MODE_MAP.items():
        if getattr(args, arg_name, False):
            mode_override, mode_flag = override, flag
            break

    # Sort by DAG stages (with mode override if CLI flag set)
    sorted_runs, group_id_map, dag_lines, deps_map = sort_pipelines(selected_pipelines, all_dags, mode_override)

    # Local mode only supports a single script
    if args.local and len(selected_pipelines) > 1:
        print(f"\n--local only supports a single script, but {len(selected_pipelines)} matched:")
        for p in selected_pipelines:
            print(f"   {p.name}")
        print("\nNarrow your selection to a single script.")
        exit(1)

    mode_name = mode_override.upper() if mode_override else "JSON defaults"

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
            deps = deps_map.get((script, mode), {})
            if deps.get("outputs"):
                cmd.extend(["--outputs", ",".join(deps["outputs"])])
            if deps.get("inputs"):
                cmd.extend(["--inputs", ",".join(deps["inputs"])])
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
