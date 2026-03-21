"""Launch ML pipelines via SQS or locally.

Run this from a directory containing pipeline subdirectories (e.g., ml_pipelines/).
Scripts can be defined in pipelines.json DAGs or run as standalone scripts.

Usage:
    ml_pipeline_launcher --dt                    # Launch 1 random pipeline group (all scripts in a directory)
    ml_pipeline_launcher --dt -n 3               # Launch 3 random pipeline groups
    ml_pipeline_launcher --dt --all              # Launch ALL pipelines
    ml_pipeline_launcher --dt caco2              # Launch pipelines matching 'caco2'
    ml_pipeline_launcher --dt caco2 ppb          # Launch pipelines matching 'caco2' or 'ppb'
    ml_pipeline_launcher my_script.py             # Launch exact script (modeless, via SQS)
    ml_pipeline_launcher caco2                    # Launch all pipelines matching 'caco2' (substring)
    ml_pipeline_launcher --promote --all         # Promote ALL pipelines
    ml_pipeline_launcher --test-promote --all    # Test-promote ALL pipelines
    ml_pipeline_launcher --ts --all              # Temporal split ALL pipelines
    ml_pipeline_launcher --dt --dry-run          # Show what would be launched without launching
    ml_pipeline_launcher --local --dt ppb_human  # Run pipelines locally (uses active Python interpreter)
    ml_pipeline_launcher --dt my_standalone      # Launch standalone script in DT mode
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

VERSION_RE = re.compile(r"^(.+?)_v?(\d+)$")
FRAMEWORK_RE = re.compile(r"-(xgb|pytorch|chemprop|chemeleon)$")

MODES = ["dt", "promote", "test_promote", "ts"]
FILTER_MODES = {"dt", "ts"}
OVERRIDE_MODES = {"promote", "test_promote"}


@dataclass
class RunPlan:
    """Execution plan produced by sort_pipelines()."""

    runs: list[tuple[Path, str]] = field(default_factory=list)
    group_ids: dict[tuple[Path, str], str | None] = field(default_factory=dict)
    display_lines: list[str] = field(default_factory=list)
    deps: dict[tuple[Path, str], dict] = field(default_factory=dict)


def parse_script_name(script_path: Path) -> tuple[str, str | None]:
    """Parse a pipeline script filename into (basename, version).

    If the filename ends with _{number} or _v{number}, extracts the version.
    Otherwise returns the full stem with no version.

    Args:
        script_path (Path): Path to the pipeline script

    Returns:
        tuple[str, str | None]: (basename, version) — e.g., ("ppb_human_free_reg_xgb", "1")
            or ("caco2_er_reg_open_admet", None) for unversioned scripts
    """
    match = VERSION_RE.match(script_path.stem)
    if match:
        return match.group(1), match.group(2)
    return script_path.stem, None


def build_pipeline_meta(script_path: Path, mode: str, serverless: bool) -> str:
    """Build PIPELINE_META JSON for a pipeline script.

    Derives model_name and endpoint_name from the script filename and mode.
    For promoted endpoints, the framework suffix (xgb, pytorch, etc.) is stripped.
    Version suffix is included in names when present in the filename.
    """
    from datetime import datetime

    basename, version = parse_script_name(script_path)
    basename_hyphen = basename.replace("_", "-")  # e.g., "ppb-human-free-reg-xgb"
    endpoint_base = FRAMEWORK_RE.sub("", basename_hyphen)  # e.g., "ppb-human-free-reg"
    today = datetime.now().strftime("%y%m%d")
    v = f"-{version}" if version else ""

    if mode in ("dt", "ts"):
        model_name = f"{basename_hyphen}{v}-{mode}"
        endpoint_name = f"{basename_hyphen}{v}-{mode}"
    elif mode == "promote":
        model_name = f"{basename_hyphen}{v}-{today}"
        endpoint_name = f"{endpoint_base}{v}"
    elif mode == "test_promote":
        model_name = f"{basename_hyphen}{v}-{today}"
        endpoint_name = f"{endpoint_base}{v}-test"
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
    mode: str | None = None,
) -> RunPlan:
    """Sort pipelines by DAG stages with per-script modes.

    Scripts defined in a pipelines.json DAG are ordered by stage. Standalone
    scripts (not in any DAG) are appended as independent runs.

    Mode behavior depends on the mode type:
        - Filter modes (dt, ts): Only run scripts that declare the mode in
          their pipelines.json entry. Respects DAG ordering.
        - Override modes (promote, test_promote): Run every unique script once
          with the override mode, ignoring JSON modes. No DAG ordering.
        - None: Run all modes from pipelines.json, respecting DAG ordering.

    Args:
        pipelines (list[Path]): Pipelines to sort
        all_dags (dict): DAG definitions from pipelines.json files
            {dag_name: [stages]} where each stage is {script_path: [modes]}
        mode (str | None): Execution mode from CLI flags (e.g., "dt", "promote")

    Returns:
        RunPlan: Execution plan with runs, group IDs, display lines, and dependencies
    """
    # Override modes (promote, test_promote) take a completely different path:
    # deduplicate scripts and run each once, no DAG ordering needed.
    if mode and mode in OVERRIDE_MODES:
        return _build_override_plan(pipelines, all_dags, mode)

    return _build_dag_plan(pipelines, all_dags, mode_filter=mode)


def _build_override_plan(
    pipelines: list[Path],
    all_dags: dict[str, list[dict[Path, list[str]]]],
    mode: str,
) -> RunPlan:
    """Build a plan that runs each unique script once with the override mode.

    Used for promote/test_promote where DAG ordering is irrelevant.

    Args:
        pipelines (list[Path]): Pipelines to run
        all_dags (dict): DAG definitions (used only to identify DAG vs standalone)
        mode (str): The override mode (e.g., "promote")

    Returns:
        RunPlan: Execution plan with one run per unique script
    """
    pipeline_set = set(pipelines)
    plan = RunPlan()
    seen_scripts = set()

    # Collect unique scripts from DAGs (in DAG order for display consistency)
    for dag_name, stages in all_dags.items():
        for stage in stages:
            for script in stage:
                if script in pipeline_set and script not in seen_scripts:
                    seen_scripts.add(script)
                    run = (script, mode)
                    plan.runs.append(run)
                    plan.group_ids[run] = None
                    plan.deps[run] = {"outputs": [], "inputs": []}
                    plan.display_lines.append(f"   {script.stem}:{mode}")

    # Add standalone scripts
    for script in pipelines:
        if script not in seen_scripts:
            run = (script, mode)
            plan.runs.append(run)
            plan.group_ids[run] = None
            plan.deps[run] = {"outputs": [], "inputs": []}
            plan.display_lines.append(f"   {script.stem}:{mode} (standalone)")

    return plan


def _build_dag_plan(
    pipelines: list[Path],
    all_dags: dict[str, list[dict[Path, list[str]]]],
    mode_filter: str | None = None,
) -> RunPlan:
    """Build a plan respecting DAG stage ordering, optionally filtering by mode.

    Args:
        pipelines (list[Path]): Pipelines to sort
        all_dags (dict): DAG definitions from pipelines.json files
        mode_filter (str | None): If set, only include runs for this mode

    Returns:
        RunPlan: Execution plan with runs in DAG stage order
    """
    pipeline_set = set(pipelines)
    plan = RunPlan()
    found_in_dag = set()

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
                found_in_dag.add(script)
                dag_has_runs = True
                if mode_filter and mode_filter not in modes:
                    continue
                for m in ([mode_filter] if mode_filter else modes):
                    run = (script, m)
                    plan.runs.append(run)
                    plan.group_ids[run] = dag_name
                    plan.deps[run] = {"outputs": outputs, "inputs": inputs}
                    stage_parts.append(f"{script.stem}:{m}")
            if stage_parts:
                dag_stage_lines.append(" | ".join(stage_parts))
        if dag_has_runs:
            plan.display_lines.append("   " + " --> ".join(dag_stage_lines))

    # Handle standalone scripts (not in any DAG)
    for script in pipelines:
        if script in found_in_dag:
            continue
        run = (script, mode_filter)
        plan.runs.append(run)
        plan.group_ids[run] = None
        plan.deps[run] = {"outputs": [], "inputs": []}
        label = f"{script.stem}:{mode_filter}" if mode_filter else script.stem
        plan.display_lines.append(f"   {label} (standalone)")

    return plan


def get_all_pipelines() -> tuple[list[Path], dict[str, list[dict[Path, list[str]]]]]:
    """Get all ML pipeline scripts from subdirectories.

    Discovers scripts from pipelines.json files AND standalone scripts
    (matching the version naming pattern) that aren't in any DAG.

    Returns:
        tuple: (pipelines, all_dags)
            - pipelines: List of unique pipeline script paths
            - all_dags: {dag_name: [stages]} from pipelines.json files
    """
    cwd = Path.cwd()
    pipelines = []
    all_dags = {}
    seen_scripts = set()

    # Discover scripts from pipelines.json files
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

    # Discover standalone scripts not in any DAG (skip __init__.py and private modules)
    for py_file in sorted(cwd.rglob("*.py")):
        if py_file in seen_scripts:
            continue
        if py_file.stem.startswith("__"):
            continue
        pipelines.append(py_file)
        seen_scripts.add(py_file)

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
    """Filter pipelines by patterns matching the basename.

    If a pattern ends with '.py', it matches the exact filename (so 'my_script.py'
    won't match 'my_script_test.py'). Otherwise, it's a case-insensitive substring
    match against the stem.
    """
    if not patterns:
        return pipelines

    exact = {p.lower().removesuffix(".py") for p in patterns if p.lower().endswith(".py")}
    substring = [p.lower() for p in patterns if not p.lower().endswith(".py")]

    results = []
    for p in pipelines:
        stem = p.stem.lower()
        if stem in exact:
            results.append(p)
        elif any(pat in stem for pat in substring):
            results.append(p)
    return results


def get_dag_scripts(all_dags: dict) -> set[Path]:
    """Get the set of all scripts referenced in DAG definitions."""
    scripts = set()
    for stages in all_dags.values():
        for stage in stages:
            scripts.update(stage.keys())
    return scripts


def select_pipelines(all_pipelines: list[Path], args: argparse.Namespace) -> tuple[list[Path], str]:
    """Select which pipelines to run based on CLI args.

    Returns:
        tuple[list[Path], str]: (selected_pipelines, human-readable selection description)
    """
    if args.patterns:
        selected = filter_pipelines_by_patterns(all_pipelines, args.patterns)
        if not selected:
            print(f"No pipelines matching patterns: {args.patterns}")
            exit(1)
        return selected, f"matching {args.patterns}"

    if args.all:
        return all_pipelines, "ALL"

    selected = select_random_groups(all_pipelines, args.num_groups)
    if not selected:
        print("No pipeline groups found")
        exit(1)
    group_names = [d.name for d in get_pipeline_groups(selected)]
    return selected, f"RANDOM {args.num_groups} group(s): {group_names}"


def resolve_mode(args: argparse.Namespace, selected: list[Path], all_dags: dict) -> str | None:
    """Determine the execution mode from CLI flags.

    Returns:
        str | None: Mode name (e.g., "dt") or None for modeless standalone execution
    """
    # Check explicit CLI flags first
    for mode_name in MODES:
        if getattr(args, mode_name, False):
            return mode_name

    # No mode flag given — that's fine for standalone scripts (they run modeless).
    # For DAG scripts, None means "use pipelines.json defaults".
    return None


def print_summary(plan: RunPlan, selection_desc: str, mode: str | None, args: argparse.Namespace, all_dags: dict):
    """Print the launch summary banner."""
    dag_scripts = get_dag_scripts(all_dags) if all_dags else set()
    has_dag_scripts = any(p in dag_scripts for p, _ in plan.runs)
    if mode:
        mode_display = mode.upper()
    elif has_dag_scripts:
        mode_display = "JSON defaults"
    else:
        mode_display = "none (standalone)"
    prefix = "DRY RUN - " if args.dry_run else ""

    print(f"\n{'=' * 60}")
    print(f"{prefix}LAUNCHING {len(plan.runs)} PIPELINE RUNS")
    print(f"{'=' * 60}")
    print(f"Source: {Path.cwd()}")
    print(f"Selection: {selection_desc}")
    print(f"Mode: {mode_display}")
    print(f"Execution: {'Local' if args.local else 'SQS → Batch'}")
    print(f"Endpoint: {'Realtime' if args.realtime else 'Serverless'}")
    print("\nPipeline DAGs:")
    for line in plan.display_lines:
        print(line)
    print()


def run_pipelines(plan: RunPlan, args: argparse.Namespace):
    """Execute all pipeline runs (local or SQS)."""
    serverless = not args.realtime

    # Countdown before launching (skip for local runs)
    if not args.local:
        print("Launching in ", end="", flush=True)
        for i in range(10, 0, -1):
            print(f"{i}...", end="", flush=True)
            time.sleep(1)
        print(" GO!\n")

    for i, (script, mode) in enumerate(plan.runs, 1):
        mode_display = f" ({mode})" if mode else ""
        print(f"\n{'─' * 60}")
        print(f"{'Running' if args.local else 'Launching'} run {i}/{len(plan.runs)}: {script.name}{mode_display}")

        # Build PIPELINE_META only when a mode is set (standalone scripts don't need it)
        pipeline_meta = build_pipeline_meta(script, mode, serverless) if mode else None

        if args.local:
            env = os.environ.copy()
            if pipeline_meta:
                env["PIPELINE_META"] = pipeline_meta
                print(f"with ENV: PIPELINE_META='{pipeline_meta}'")
            print(f"{'─' * 60}\n")
            cmd = [sys.executable, str(script)]
            result = subprocess.run(cmd, env=env)
        else:
            cmd = ["ml_pipeline_sqs", str(script)]
            if mode:
                cmd.append(f"--{mode.replace('_', '-')}")
            if args.realtime:
                cmd.append("--realtime")
            if pipeline_meta:
                cmd.extend(["--pipeline-meta", pipeline_meta])
            group_id = plan.group_ids.get((script, mode))
            if group_id:
                cmd.extend(["--group-id", group_id])
            deps = plan.deps.get((script, mode), {})
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
    print(f"FINISHED LAUNCHING {len(plan.runs)} PIPELINE RUNS")
    print(f"{'=' * 60}\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch ML pipelines via SQS or locally")
    parser.add_argument(
        "patterns",
        nargs="*",
        help="Filter patterns: 'foo.py' matches exactly, 'foo' matches as substring (e.g., 'caco2' 'ppb')",
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
    mode_group.add_argument("--ts", action="store_true", help="Override all scripts to temporal split evaluation mode")

    return parser.parse_args()


def main():
    args = parse_args()

    # Discover all pipelines and DAG definitions
    all_pipelines, all_dags = get_all_pipelines()
    if not all_pipelines:
        print(f"No pipeline scripts found in subdirectories of {Path.cwd()}")
        exit(1)

    # Select which pipelines to run
    selected, selection_desc = select_pipelines(all_pipelines, args)

    # Validate --local before doing more work
    if args.local and len(selected) > 1:
        print(f"\n--local only supports a single script, but {len(selected)} matched:")
        for p in selected:
            print(f"   {p.name}")
        print("\nNarrow your selection to a single script.")
        exit(1)

    # Resolve execution mode
    mode = resolve_mode(args, selected, all_dags)

    # Build execution plan
    plan = sort_pipelines(selected, all_dags, mode)

    # Display summary
    print_summary(plan, selection_desc, mode, args, all_dags)

    # Execute (unless dry run)
    if args.dry_run:
        print("Dry run complete. No pipelines were launched.\n")
        return

    run_pipelines(plan, args)


if __name__ == "__main__":
    main()
