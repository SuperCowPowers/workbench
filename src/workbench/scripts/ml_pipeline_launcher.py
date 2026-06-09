"""Launch ML pipelines via SQS or locally.

Run this from a directory containing pipeline subdirectories (e.g., ml_pipelines/).
Scripts can be defined in pipelines.json DAGs or run as standalone scripts.

Usage:
    ml_pipeline_launcher --dt --all              # Launch ALL pipelines in DT mode
    ml_pipeline_launcher --dt caco2              # Launch pipelines matching 'caco2' in DT mode
    ml_pipeline_launcher --dt caco2 ppb          # Launch pipelines matching 'caco2' or 'ppb'
    ml_pipeline_launcher --ts --all              # Temporal split ALL pipelines
    ml_pipeline_launcher --promote --all         # Promote ALL pipelines
    ml_pipeline_launcher caco2                   # Launch all pipelines matching 'caco2' (JSON default modes)
    ml_pipeline_launcher my_script.py            # Launch exact script (modeless, via SQS)
    ml_pipeline_launcher --dt --dry-run          # Show what would be launched without launching
    ml_pipeline_launcher --local --dt ppb_human  # Run a single pipeline locally

Args after a literal '--' are forwarded verbatim to the underlying script:
    ml_pipeline_launcher --dt my_script.py -- --epochs 10 --lr 0.01
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from workbench.utils.repl_utils import colors as REPL_COLORS

VERSION_RE = re.compile(r"^(.+?)_v?(\d+)$")
FRAMEWORK_RE = re.compile(r"-(xgb|pytorch|chemprop|chemeleon)$")

FILTER_MODES = {"dt", "ts"}
OVERRIDE_MODES = {"promote"}
ALL_MODES = FILTER_MODES | OVERRIDE_MODES

MODE_COLORS = {"dt": "lightgreen", "ts": "darkyellow"}


def run_label(script: Path, mode: str | None) -> str:
    """Human-readable label for a (script, mode) run: 'script [mode]' or 'script'."""
    return f"{script.stem} [{mode}]" if mode else script.stem


def _color(text: str, color: str) -> str:
    """Wrap text in an ANSI color from the repl palette (no-op when stdout isn't a TTY)."""
    if not sys.stdout.isatty():
        return text
    return f"{REPL_COLORS[color]}{text}{REPL_COLORS['reset']}"


def display_label(script: Path, mode: str | None, leaf: bool = False) -> str:
    """run_label colorized for terminal display.

    The mode bracket is tinted by mode; leaf nodes (terminal consumers) get a
    subtle blue name to set them apart from producers/intermediate nodes.
    """
    stem = _color(script.stem, "lightblue") if leaf else script.stem
    if not mode:
        return stem
    return f"{stem} {_color(f'[{mode}]', MODE_COLORS.get(mode, 'lightgrey'))}"


@dataclass
class PipelineNode:
    """A single node in a pipeline DAG: a script run in an optional mode.

    A node declares the artifacts it outputs and inputs (refs of the form
    "type:name", e.g. "fs:aqsol_features"). DAG edges are derived by matching a
    node's input artifacts to whichever node outputs them, so the same script
    can appear as multiple nodes under different modes.

    Attributes:
        script (Path): Path to the pipeline script
        mode (str | None): Execution mode ("dt"/"ts"), or None for a modeless node
        outputs (list[str]): Artifact refs this node produces
        inputs (list[str]): Artifact refs this node depends on
    """

    script: Path
    mode: str | None = None
    outputs: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        """Stable label for this node, unique within a pipeline."""
        return run_label(self.script, self.mode)


@dataclass
class RunPlan:
    """Execution plan produced by sort_pipelines()."""

    runs: list[tuple[Path, str]] = field(default_factory=list)
    group_ids: dict[tuple[Path, str], str | None] = field(default_factory=dict)
    display_lines: list[str] = field(default_factory=list)
    deps: dict[tuple[Path, str], dict] = field(default_factory=dict)

    def add_run(
        self,
        script: Path,
        mode: str,
        group_id: str | None = None,
        outputs: list[str] | None = None,
        inputs: list[str] | None = None,
    ):
        """Add a run to the plan.

        Args:
            script (Path): Path to the pipeline script
            mode (str): Execution mode (e.g., "dt", "promote")
            group_id (str | None): Pipeline name for grouped execution, or None for standalone
            outputs (list[str] | None): Artifact refs this run produces
            inputs (list[str] | None): Artifact refs this run consumes (its dependencies)
        """
        run = (script, mode)
        self.runs.append(run)
        self.group_ids[run] = group_id
        self.deps[run] = {"outputs": outputs or [], "inputs": inputs or []}


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
    else:
        # Non-model mode (e.g. "fs" FeatureSet producers, or future modes): nothing
        # to name, so pass the mode through with no model/endpoint metadata. Warn so
        # a typo'd mode is still visible rather than silently producing no model.
        print(
            f"NOTE: mode {mode!r} is not a model-producing mode; passing through with no model metadata.",
            file=sys.stderr,
        )
        return json.dumps({"mode": mode, "serverless": serverless})

    return json.dumps(
        {"mode": mode, "model_name": model_name, "endpoint_name": endpoint_name, "serverless": serverless}
    )


def load_pipelines_config(directory: Path) -> dict[str, list[PipelineNode]] | None:
    """Load pipelines.json from a directory.

    The JSON maps each pipeline name to a flat list of nodes. A node runs a
    script in an optional mode and declares the artifacts it outputs/inputs.
    DAG edges, and therefore execution order, are derived from those artifacts.

    Args:
        directory (Path): Directory to check for pipelines.json

    Returns:
        dict | None: {pipeline_name: [PipelineNode]}, or None if no JSON config found
    """
    json_path = directory / "pipelines.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        config = json.load(f)
    pipelines = {}
    for name, raw_nodes in config.get("pipelines", {}).items():
        pipelines[name] = [
            PipelineNode(
                script=directory / raw["script"],
                mode=raw.get("mode"),
                outputs=raw.get("outputs", []),
                inputs=raw.get("inputs", []),
            )
            for raw in raw_nodes
        ]
    return pipelines


def build_dag(name: str, nodes: list[PipelineNode]) -> nx.DiGraph:
    """Build a DiGraph for a pipeline, deriving edges from node outputs/inputs.

    Each node becomes a graph node keyed by (script, mode), with the
    PipelineNode stored under the "node" attribute. Edges run producer ->
    consumer, matched by artifact ref.

    Args:
        name (str): Pipeline name (used in error/warning messages)
        nodes (list[PipelineNode]): Nodes belonging to the pipeline

    Returns:
        nx.DiGraph: The derived dependency graph

    Raises:
        ValueError: If two nodes output the same artifact, if a node is
            duplicated, or if the resulting graph has a dependency cycle.
    """
    graph = nx.DiGraph()
    producer_of: dict[str, tuple[Path, str | None]] = {}

    # Register nodes and index artifact producers (one producer per artifact)
    for node in nodes:
        key = (node.script, node.mode)
        if key in graph:
            raise ValueError(f"Pipeline '{name}': duplicate node {node.node_id}")
        graph.add_node(key, node=node)
        for artifact in node.outputs:
            if artifact in producer_of:
                other = run_label(*producer_of[artifact])
                raise ValueError(
                    f"Pipeline '{name}': artifact '{artifact}' is output by "
                    f"both {other} and {node.node_id} (each artifact needs exactly one producer)"
                )
            producer_of[artifact] = key

    # Derive edges: each input artifact links back to its producer
    for node in nodes:
        for artifact in node.inputs:
            producer = producer_of.get(artifact)
            if producer is None:
                print(
                    f"WARNING: pipeline '{name}': node {node.node_id} inputs '{artifact}', "
                    f"which no node in the pipeline outputs; treating it as an external input.",
                    file=sys.stderr,
                )
                continue
            graph.add_edge(producer, (node.script, node.mode))

    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        readable = " -> ".join(run_label(s, m) for (s, m), _ in cycle)
        raise ValueError(f"Pipeline '{name}' has a dependency cycle: {readable}")

    return graph


def sort_pipelines(
    pipelines: list[Path],
    all_dags: dict[str, list[PipelineNode]],
    mode: str | None = None,
) -> RunPlan:
    """Sort pipelines into a topologically ordered run plan.

    Scripts defined in a pipelines.json DAG are ordered by their derived
    artifact dependencies. Standalone scripts (not in any DAG) are appended as
    independent runs.

    Mode behavior depends on the mode type:
        - Filter modes (dt, ts): Only run nodes whose mode matches (modeless
          nodes always run). Respects DAG ordering.
        - Override modes (promote): Run every unique script once
          with the override mode, ignoring node modes. No DAG ordering.
        - None: Run all nodes (every mode), respecting DAG ordering.

    Args:
        pipelines (list[Path]): Pipelines to sort
        all_dags (dict): DAG definitions from pipelines.json files
            {pipeline_name: [PipelineNode]}
        mode (str | None): Execution mode from CLI flags (e.g., "dt", "promote")

    Returns:
        RunPlan: Execution plan with runs, group IDs, display lines, and dependencies
    """
    # Override modes (promote) take a completely different path:
    # deduplicate scripts and run each once, no DAG ordering needed.
    if mode and mode in OVERRIDE_MODES:
        return _build_override_plan(pipelines, all_dags, mode)

    return _build_dag_plan(pipelines, all_dags, mode_filter=mode)


def _build_override_plan(
    pipelines: list[Path],
    all_dags: dict[str, list[PipelineNode]],
    mode: str,
) -> RunPlan:
    """Build a plan that runs each unique script once with the override mode.

    Used for promote where DAG ordering is irrelevant.

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

    # Collect unique scripts from DAGs (in node order for display consistency)
    for nodes in all_dags.values():
        for node in nodes:
            script = node.script
            if script in pipeline_set and script not in seen_scripts:
                seen_scripts.add(script)
                plan.add_run(script, mode)
                plan.display_lines.append(f"   {run_label(script, mode)}")

    # Add standalone scripts
    for script in pipelines:
        if script not in seen_scripts:
            plan.add_run(script, mode)
            plan.display_lines.append(f"   {run_label(script, mode)} (standalone)")

    return plan


def _node_selected(node: PipelineNode, mode_filter: str | None) -> bool:
    """Whether a node runs under the given filter mode.

    A filter (e.g. ``--dt``) only excludes *other filter modes*. Modeless nodes,
    and named non-filter modes like ``fs`` producers, are prerequisites that run
    under every filter -- they're never selected away.
    """
    if mode_filter is None or node.mode not in FILTER_MODES:
        return True
    return node.mode == mode_filter


def _build_dag_plan(
    pipelines: list[Path],
    all_dags: dict[str, list[PipelineNode]],
    mode_filter: str | None = None,
) -> RunPlan:
    """Build a plan in topological order, optionally filtering by mode.

    Edges are derived from each pipeline's outputs/inputs artifacts. The
    full (all-mode) graph is built so cross-mode producers resolve, then only
    the nodes selected by ``mode_filter`` are emitted, preserving topological
    order. A node's produced/consumed artifacts become its outputs/inputs for
    the batch dependency layer.

    Args:
        pipelines (list[Path]): Pipelines to sort
        all_dags (dict): DAG definitions from pipelines.json files
        mode_filter (str | None): If set, only include runs for this mode

    Returns:
        RunPlan: Execution plan with runs in topological order
    """
    pipeline_set = set(pipelines)
    plan = RunPlan()
    found_in_dag = set()

    for name, nodes in all_dags.items():
        in_pipeline = [n for n in nodes if n.script in pipeline_set]
        found_in_dag.update(n.script for n in in_pipeline)
        if not in_pipeline:
            continue

        graph = build_dag(name, in_pipeline)

        # Emit runs in topological order; collect the selected keys for display.
        selected = []
        for generation in nx.topological_generations(graph):
            for key in generation:
                node = graph.nodes[key]["node"]
                if not _node_selected(node, mode_filter):
                    continue
                # A named non-filter mode (e.g. "fs") runs as itself; a modeless
                # node adopts the active filter; a filter mode keeps its mode.
                if node.mode and node.mode not in FILTER_MODES:
                    runtime_mode = node.mode
                else:
                    runtime_mode = mode_filter or node.mode
                plan.add_run(
                    node.script,
                    runtime_mode,
                    group_id=name,
                    outputs=node.outputs,
                    inputs=node.inputs,
                )
                selected.append((key, node, runtime_mode))
        if not selected:
            continue

        # Render the selected sub-DAG as a Unicode tree (producers -> consumers).
        subgraph = graph.subgraph(key for key, _, _ in selected).copy()
        for key, node, runtime_mode in selected:
            leaf = subgraph.out_degree(key) == 0
            subgraph.nodes[key]["label"] = display_label(node.script, runtime_mode, leaf=leaf)
        plan.display_lines.append(f"   {_color(name, 'lightpurple')}")
        for line in nx.generate_network_text(subgraph, with_labels=True):
            plan.display_lines.append(f"   {line}")

    # Handle standalone scripts (not in any DAG)
    for script in pipelines:
        if script in found_in_dag:
            continue
        plan.add_run(script, mode_filter)
        plan.display_lines.append(f"   {run_label(script, mode_filter)} (standalone)")

    return plan


def get_all_pipelines() -> tuple[list[Path], dict[str, list[PipelineNode]]]:
    """Get all ML pipeline scripts from subdirectories.

    Discovers scripts from pipelines.json files AND standalone scripts
    (matching the version naming pattern) that aren't in any DAG.

    Returns:
        tuple: (pipelines, all_dags)
            - pipelines: List of unique pipeline script paths
            - all_dags: {pipeline_name: [PipelineNode]} from pipelines.json files
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
            for nodes in dag_defs.values():
                for node in nodes:
                    if node.script not in seen_scripts:
                        pipelines.append(node.script)
                        seen_scripts.add(node.script)

    # Discover standalone scripts not in any DAG (skip __init__.py and private modules)
    for py_file in sorted(cwd.rglob("*.py")):
        if py_file in seen_scripts:
            continue
        if py_file.stem.startswith("__"):
            continue
        pipelines.append(py_file)
        seen_scripts.add(py_file)

    return pipelines, all_dags


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

    print("Specify pipeline patterns or use --all to launch all pipelines.")
    exit(1)


def resolve_mode(args: argparse.Namespace) -> str | None:
    """Determine the execution mode from CLI flags.

    Returns:
        str | None: Mode name (e.g., "dt") or None for default JSON modes
    """
    for mode_name in ALL_MODES:
        if getattr(args, mode_name, False):
            return mode_name
    return None


def print_summary(plan: RunPlan, selection_desc: str, mode: str | None, args: argparse.Namespace):
    """Print the launch summary banner."""
    if mode:
        mode_display = mode.upper()
    elif any(gid is not None for gid in plan.group_ids.values()):
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


def run_pipelines(plan: RunPlan, args: argparse.Namespace, extra_args: list[str]):
    """Execute all pipeline runs (local or SQS).

    Args:
        plan (RunPlan): Execution plan from sort_pipelines()
        args (argparse.Namespace): Parsed launcher arguments
        extra_args (list[str]): Args after the ``--`` separator, forwarded
            verbatim to each underlying pipeline script
    """
    serverless = not args.realtime

    # Countdown before launching (skip for local runs)
    if not args.local:
        print("Launching in ", end="", flush=True)
        for i in range(10, 0, -1):
            print(f"{i}...", end="", flush=True)
            time.sleep(1)
        print(" GO!\n")

    for i, (script, mode) in enumerate(plan.runs, 1):
        mode_display = f" [{mode}]" if mode else ""
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
            cmd = [sys.executable, str(script), *extra_args]
            result = subprocess.run(cmd, env=env)
        else:
            cmd = ["ml_pipeline_sqs", str(script)]
            if mode:
                cmd.append(f"--{mode.replace('_', '-')}")
            if args.realtime:
                cmd.append("--realtime")
            if pipeline_meta:
                cmd.extend(["--pipeline-meta", pipeline_meta])
            if extra_args:
                cmd.extend(["--script-args", json.dumps(extra_args)])
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


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command-line arguments.

    A literal ``--`` separates launcher args from args meant for the underlying
    pipeline script. Everything after ``--`` is left unparsed and forwarded
    verbatim to the script (locally or via SQS → Batch).

    Returns:
        tuple[argparse.Namespace, list[str]]: Parsed launcher args, and the list
            of args to forward to the script (empty if no ``--`` is present).
    """
    argv = sys.argv[1:]
    extra_args: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        argv, extra_args = argv[:idx], argv[idx + 1 :]

    parser = argparse.ArgumentParser(description="Launch ML pipelines via SQS or locally")
    parser.add_argument(
        "patterns",
        nargs="*",
        help="Filter patterns: 'foo.py' matches exactly, 'foo' matches as substring (e.g., 'caco2' 'ppb')",
    )
    parser.add_argument("--all", action="store_true", help="Launch ALL pipelines")
    parser.add_argument("--realtime", action="store_true", help="Create realtime endpoints (default is serverless)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be launched without actually launching")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run pipelines locally instead of via SQS (uses active Python interpreter)",
    )

    # Mode flags (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dt", action="store_true", help="Filter to DT mode (dynamic training)")
    mode_group.add_argument("--ts", action="store_true", help="Filter to temporal split evaluation mode")
    mode_group.add_argument("--promote", action="store_true", help="Run all scripts in PROMOTE mode")

    return parser.parse_args(argv), extra_args


def main():
    args, extra_args = parse_args()

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
    mode = resolve_mode(args)

    # Build execution plan
    try:
        plan = sort_pipelines(selected, all_dags, mode)
    except ValueError as e:
        print(f"\nERROR: {e}")
        exit(1)

    # Display summary
    print_summary(plan, selection_desc, mode, args)
    if extra_args:
        print(f"Forwarding to script: {' '.join(extra_args)}\n")

    # Execute (unless dry run)
    if args.dry_run:
        print("Dry run complete. No pipelines were launched.\n")
        return

    run_pipelines(plan, args, extra_args)


if __name__ == "__main__":
    main()
