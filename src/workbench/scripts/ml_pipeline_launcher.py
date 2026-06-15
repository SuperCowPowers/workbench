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
import contextlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from workbench.utils.repl_utils import colors as REPL_COLORS
from workbench.utils.tree_render import render_forest
from workbench.lambda_layer.pipeline_manager import (
    Job,
    PipelineManager,
    parse_spec,
    ref_type,
    simulated_mtime,
)

log = logging.getLogger("workbench")

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


def script_label(stem: str, mode: str | None) -> str:
    """Data-flow label for a script node: blue name + mode-colored [mode] bracket."""
    name = _color(stem, "lightblue")
    if not mode:
        return name
    return f"{name} {_color(f'[{mode}]', MODE_COLORS.get(mode, 'lightgrey'))}"


@dataclass
class RunPlan:
    """Execution plan from sort_pipelines(): the run-jobs plus the display lines.

    Each run is a :class:`Job` whose ``mode`` is the *runtime* mode (a modeless
    node adopts the active filter). ``group``/``outputs``/``inputs`` ride along on
    the job, so there are no parallel dicts to keep in sync.
    """

    runs: list[Job] = field(default_factory=list)
    display_lines: list[str] = field(default_factory=list)
    environment_suspect: bool = False  # nearly every artifact missing -> likely wrong account/region

    def add_run(self, job: Job):
        """Add a run-job to the plan."""
        self.runs.append(job)


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


def build_pipeline_meta(job: Job, serverless: bool) -> str:
    """Build the PIPELINE_META JSON for a run.

    Declared dt/ts model jobs use the shared core (:meth:`Job.pipeline_meta` --
    model/endpoint names from the declared ``model:`` output). The launcher adds
    two cases the core can't express: ``promote`` derives a date-stamped name from
    the filename (framework suffix stripped for the endpoint), and a
    standalone/undeclared dt/ts run falls back to a filename-derived name.
    """
    from datetime import datetime

    mode = job.mode
    model_ref = next((o for o in job.outputs if ref_type(o) == "model"), None)

    # Declared dt/ts with a model: output -> the shared core.
    if mode in ("dt", "ts") and model_ref is not None:
        return json.dumps(job.pipeline_meta(serverless))

    basename, version = parse_script_name(Path(job.script))
    basename_hyphen = basename.replace("_", "-")  # e.g., "ppb-human-free-reg-xgb"
    v = f"-{version}" if version else ""

    if mode in ("dt", "ts"):  # no declared output -> filename fallback
        name = f"{basename_hyphen}{v}-{mode}"
        return json.dumps({"serverless": serverless, "mode": mode, "model_name": name, "endpoint_name": name})

    if mode == "promote":
        endpoint_base = FRAMEWORK_RE.sub("", basename_hyphen)  # e.g., "ppb-human-free-reg"
        name = f"{basename_hyphen}{v}-{datetime.now().strftime('%y%m%d')}"
        return json.dumps(
            {"serverless": serverless, "mode": mode, "model_name": name, "endpoint_name": f"{endpoint_base}{v}"}
        )

    # Modeless producer (e.g. FeatureSet) -> no model. A non-empty unrecognized
    # mode is likely a typo, so warn.
    if mode:
        print(
            f"NOTE: unrecognized mode {mode!r} (expected dt/ts/promote); passing through with no model metadata.",
            file=sys.stderr,
        )
    return json.dumps({"mode": mode, "serverless": serverless})


def load_pipelines_config(directory: Path) -> dict[str, list[Job]] | None:
    """Load pipelines.json from a directory.

    The JSON maps each pipeline name to a flat list of nodes. A node runs a
    script in an optional mode and declares the artifacts it outputs/inputs.
    DAG edges, and therefore execution order, are derived from those artifacts.

    Args:
        directory (Path): Directory to check for pipelines.json

    Returns:
        dict | None: {pipeline_name: [Job]}, or None if no JSON config found
    """
    json_path = directory / "pipelines.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        config = json.load(f)
    # Resolve each node's script relative to this config's directory, then group
    # the flat node list back into {pipeline_name: [nodes]}.
    nodes = parse_spec(config, script_resolver=lambda script: directory / script)
    pipelines: dict[str, list[Job]] = {}
    for node in nodes:
        pipelines.setdefault(node.group, []).append(node)
    return pipelines


def sort_pipelines(
    pipelines: list[Path],
    all_dags: dict[str, list[Job]],
    mode: str | None = None,
    selected_keys: set | None = None,
    mtime_fn=None,
    full_dag: bool = False,
    session=None,
    force_keys: set | None = None,
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
            {pipeline_name: [Job]}
        mode (str | None): Execution mode from CLI flags (e.g., "dt", "promote")

    Returns:
        RunPlan: Execution plan with runs, group IDs, display lines, and dependencies
    """
    # Override modes (promote) take a completely different path:
    # deduplicate scripts and run each once, no DAG ordering needed.
    if mode and mode in OVERRIDE_MODES:
        return _build_override_plan(pipelines, all_dags, mode)

    return _build_dag_plan(
        pipelines,
        all_dags,
        mode_filter=mode,
        selected_keys=selected_keys,
        mtime_fn=mtime_fn,
        full_dag=full_dag,
        session=session,
        force_keys=force_keys,
    )


def _build_override_plan(
    pipelines: list[Path],
    all_dags: dict[str, list[Job]],
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
                plan.add_run(Job(script, mode))
                plan.display_lines.append(f"   {run_label(script, mode)}")

    # Add standalone scripts
    for script in pipelines:
        if script not in seen_scripts:
            plan.add_run(Job(script, mode))
            plan.display_lines.append(f"   {run_label(script, mode)} (standalone)")

    return plan


def _node_selected(node: Job, mode_filter: str | None) -> bool:
    """Whether a node runs under the given filter mode.

    A filter (e.g. ``--dt``) only excludes *other filter modes*. Modeless nodes,
    and named non-filter modes like ``fs`` producers, are prerequisites that run
    under every filter -- they're never selected away.
    """
    if mode_filter is None or node.mode not in FILTER_MODES:
        return True
    return node.mode == mode_filter


def render_dataflow(nodes: list[Job], runtime_mode, produced, roots=None, highlight=()) -> list[str]:
    """Render the data-flow DAG: ds:/fs:/model: artifacts and scripts as nodes,
    edges flowing artifact -> script -> artifact. Shared producers fan out and
    cross-pipeline joins show as multi-parent, rooted at the external sources.

    Artifacts are colored by resolution against ``produced`` (every ref any
    pipeline in the repo produces, not just the selected ones): in it -> green,
    a true external source (DataSource / static FeatureSet) -> grey. So an input
    resolves to green even when its producer isn't in this selection. Scripts
    keep their mode coloring.

    Args:
        nodes (list[Job]): The selected nodes to render.
        runtime_mode (callable): node -> the mode to display it under.
        produced (set[str]): Every artifact ref produced anywhere in the repo.
        roots (list | None): Render these refs/keys as the roots (e.g. simulated
            sources) instead of the natural in-degree-0 sources.
        highlight (iterable): Artifact refs to mark in orange (e.g. the simulated
            modified sources).
    """
    # Build the bipartite data-flow graph: artifacts and scripts as nodes, with
    # children ordered (script -> its outputs; artifact -> its consumer scripts).
    label: dict = {}
    children: dict = {}
    has_parent = set()
    order = []
    external: dict = {}  # artifact ref -> True if no selected node produces it

    def ensure(key):
        if key not in children:
            children[key] = []
            order.append(key)

    # Scripts: name blue, the [mode] bracket keeps its mode color.
    for node in nodes:
        ensure(node.key)
        label[node.key] = script_label(node.stem, runtime_mode(node))

    for node in nodes:
        for ref in node.inputs:
            ensure(ref)
            external.setdefault(ref, ref not in produced)
            children[ref].append(node.key)
            has_parent.add(node.key)
        for ref in node.outputs:
            ensure(ref)
            external.setdefault(ref, ref not in produced)
            children[node.key].append(ref)
            has_parent.add(ref)

    # Color artifacts by role: only those consumed as an input get color -- green
    # if resolved (produced by a selected node), left default ("white") if
    # external. Terminal outputs (e.g. model:, no consumer) get no color.
    # Highlighted refs (e.g. simulated modified sources) override to orange.
    highlight = set(highlight)
    for ref in external:
        if ref in highlight:
            label[ref] = _color(ref, "orange")
        else:
            consumed = bool(children[ref])
            label[ref] = _color(ref, "lightgreen") if (consumed and not external[ref]) else ref

    if roots is None:
        # Roots = external sources (in-degree 0): artifacts no one produces, plus
        # scripts with no inputs. Sorted artifacts-first by name for stable output.
        roots = sorted(
            (k for k in order if k not in has_parent),
            key=lambda k: (0 if isinstance(k, str) else 1, str(k)),
        )
    else:
        roots = [r for r in roots if r in children]  # keep only refs present in the graph
    return render_forest(roots, children, label)


def _build_dag_plan(
    pipelines: list[Path],
    all_dags: dict[str, list[Job]],
    mode_filter: str | None = None,
    selected_keys: set | None = None,
    mtime_fn=None,
    full_dag: bool = False,
    session=None,
    force_keys: set | None = None,
) -> RunPlan:
    """Build a plan of the stale jobs to run, in dependency order.

    Edges are derived from each pipeline's outputs/inputs artifacts. The full
    (all-mode) graph is built so cross-mode producers resolve, then freshness
    decides what runs: a job submits only when stale (forward flood). A node's
    produced/consumed artifacts become its outputs/inputs for the batch
    dependency layer.

    Args:
        pipelines (list[Path]): Pipelines to sort
        all_dags (dict): DAG definitions from pipelines.json files
        mode_filter (str | None): If set, only include runs for this mode
        selected_keys (set | None): If set, restrict to these ``(script, mode)``
            job keys -- so a dependency contributes only its needed mode, not every
            mode of its script. When None, every mode of every ``pipelines`` script
            is eligible (used by ``--all`` and direct callers/tests).
        mtime_fn (callable | None): Freshness clock; None -> real AWS mtimes.
        full_dag (bool): Display the whole selected closure (up-to-date nodes
            included) instead of just the will-run paths.
        force_keys (set | None): Job keys to run regardless of freshness (the
            pattern-matched scripts). Their downstream consumers flood as usual.

    Returns:
        RunPlan: Execution plan with the stale runs in topological order
    """
    pipeline_set = set(pipelines)
    plan = RunPlan()
    found_in_dag = set()

    def in_selection(node: Job) -> bool:
        return node.key in selected_keys if selected_keys is not None else node.script in pipeline_set

    def runtime_mode(node: Job) -> str | None:
        # A named non-filter mode (e.g. "fs") runs as itself; a modeless node
        # adopts the active filter; a filter mode keeps its own mode.
        if node.mode and node.mode not in FILTER_MODES:
            return node.mode
        return mode_filter or node.mode

    # Gather the selected nodes from every pipeline, tagging each with its
    # pipeline so the file-wide graph can be split back into per-pipeline trees.
    grouped: dict[str, list[Job]] = {}
    for name, nodes in all_dags.items():
        in_pipeline = [n for n in nodes if in_selection(n)]
        found_in_dag.update(n.script for n in in_pipeline)
        if not in_pipeline:
            continue
        for n in in_pipeline:
            n.group = name
        grouped[name] = in_pipeline

    if grouped:
        # One file-wide graph so an input produced by a *sibling* pipeline (e.g.
        # ppb_mt consuming a FeatureSet built by ppb_human_free) resolves to a
        # real edge instead of being mistaken for an external input.
        all_nodes = [n for nodes in grouped.values() for n in nodes]
        pm = PipelineManager.from_jobs(all_nodes, session=session)

        # Pattern-matched jobs run regardless of freshness (the user just edited them);
        # restrict to the active mode so e.g. `--dt` doesn't also force the [ts] sibling.
        forced = set()
        if force_keys:
            forced = {n.key for n in all_nodes if n.key in force_keys and _node_selected(n, mode_filter)}

        # Freshness decides the rest: a dependency submits only when stale (mtime_fn
        # None -> real AWS mtimes). Emit in global topological order so every
        # producer precedes its consumers, honoring the mode filter.
        decisions = pm._plan(mtime_fn, force=forced)
        plan.environment_suspect = pm._environment_looks_wrong()
        mode_jobs = [job for job, _, _ in decisions if _node_selected(job, mode_filter)]
        will_run = []
        for job, should_run, _reason in decisions:
            if not _node_selected(job, mode_filter) or not should_run:
                continue
            plan.add_run(replace(job, mode=runtime_mode(job)))  # run-job carries the runtime mode
            will_run.append(job)

        # Render the global data-flow DAG (artifacts + scripts). Resolution is
        # judged against every pipeline in the repo, so an input stays green even
        # when its producer isn't selected. Default shows only the will-run paths;
        # --full-dag shows the whole selected closure for diagnosing dependencies.
        produced = {ref for nodes in all_dags.values() for n in nodes for ref in n.outputs}
        display_nodes = mode_jobs if full_dag else will_run
        plan.display_lines.extend(render_dataflow(display_nodes, runtime_mode, produced))

    # Handle standalone scripts (not in any DAG)
    for script in pipelines:
        if script in found_in_dag:
            continue
        plan.add_run(Job(script, mode_filter))
        plan.display_lines.append(f"   {run_label(script, mode_filter)} (standalone)")

    return plan


def get_all_pipelines() -> tuple[list[Path], dict[str, list[Job]]]:
    """Get all ML pipeline scripts from subdirectories.

    Discovers scripts from pipelines.json files AND standalone scripts
    (matching the version naming pattern) that aren't in any DAG.

    Returns:
        tuple: (pipelines, all_dags)
            - pipelines: List of unique pipeline script paths
            - all_dags: {pipeline_name: [Job]} from pipelines.json files
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


def select_pipelines(
    all_pipelines: list[Path], all_dags: dict[str, list[Job]], args: argparse.Namespace
) -> tuple[list[Path], set | None, set | None, str]:
    """Select which pipelines to run based on CLI args.

    - ``--all`` runs every script declared in a ``pipelines.json`` DAG. Standalone
      scripts (not in any DAG) are skipped — to be run by ``--all``, declare it.
    - ``--patterns`` matches against all discovered scripts (declared *and*
      standalone) and pulls in each match's transitive upstream producers (via
      ``PipelineManager._select``), so the dependencies of a matched pipeline run first.

    Returns:
        tuple: (selected_scripts, selected_keys, force_keys, description).
        ``selected_keys`` is the precise set of ``(script, mode)`` job keys to run --
        a matched script contributes all its modes, but a dependency only the
        ``(script, mode)`` actually in the closure (so a pulled-in ``logd [dt]``
        producer doesn't also drag along its unrelated ``logd [ts]`` sibling).
        ``force_keys`` is the subset that ran because the *user named it* (the matched
        scripts, not their dependencies) -- those run regardless of freshness, since
        naming a pattern means "run what I just edited." Both are ``None`` for
        ``--all`` (run every mode of every declared script; freshness decides).
    """
    all_nodes = [node for nodes in all_dags.values() for node in nodes]
    declared = {node.script for node in all_nodes}

    if args.patterns:
        matched = filter_pipelines_by_patterns(all_pipelines, args.patterns)
        if not matched:
            print(f"No pipelines matching patterns: {args.patterns}")
            exit(1)
        matched_set = set(matched)
        # All modes of matched scripts are targets; the closure adds each target's
        # transitive upstream producer *jobs* (specific (script, mode)).
        target_nodes = [n for n in all_nodes if n.script in matched_set]
        closure = PipelineManager.from_jobs(all_nodes)._select(target_nodes)
        selected_keys = {n.key for n in closure}
        force_keys = {n.key for n in target_nodes}  # matched scripts run regardless of freshness
        wanted = matched_set | {n.script for n in closure}  # matched_set keeps loose (non-DAG) matches
        selected = [p for p in all_pipelines if p in wanted]  # preserve discovery order
        return selected, selected_keys, force_keys, f"matching {args.patterns} (+ dependencies)"

    if args.all:
        selected = [p for p in all_pipelines if p in declared]  # declared DAG scripts only
        return selected, None, None, "ALL"

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
    elif any(job.group is not None for job in plan.runs):
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

    for i, job in enumerate(plan.runs, 1):
        script, mode = job.script, job.mode
        mode_display = f" [{mode}]" if mode else ""
        print(f"\n{'─' * 60}")
        print(f"{'Running' if args.local else 'Launching'} run {i}/{len(plan.runs)}: {script.name}{mode_display}")

        # Every run gets PIPELINE_META, uniformly -- the script decides whether to
        # use it. dt/ts take the model name from the declared model: output (DAG is
        # the source of truth); see build_pipeline_meta.
        pipeline_meta = build_pipeline_meta(job, serverless)

        if args.local:
            env = os.environ.copy()
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
            cmd.extend(["--pipeline-meta", pipeline_meta])
            if extra_args:
                cmd.extend(["--script-args", json.dumps(extra_args)])
            if job.group:
                cmd.extend(["--group-id", job.group])
            if job.outputs:
                cmd.extend(["--outputs", ",".join(job.outputs)])
            if job.inputs:
                cmd.extend(["--inputs", ",".join(job.inputs)])
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
        "--full-dag",
        action="store_true",
        help="Show the whole selected dependency closure (incl. up-to-date nodes), not just what will run",
    )
    parser.add_argument(
        "--sim-mod",
        nargs="+",
        metavar="REF",
        help="Simulate these artifact refs as freshly modified; show the global DAG paths that "
        "would be submitted to Batch (no AWS, no launch). E.g. --sim-mod ds:ppb_human_assay_processed_ds",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run pipelines locally instead of via SQS (uses active Python interpreter)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Submit even when nearly every artifact is missing (otherwise treated as a likely "
        "wrong-account/region misconfig and aborted)",
    )

    # Mode flags (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dt", action="store_true", help="Filter to DT mode (dynamic training)")
    mode_group.add_argument("--ts", action="store_true", help="Filter to temporal split evaluation mode")
    mode_group.add_argument("--promote", action="store_true", help="Run all scripts in PROMOTE mode")

    return parser.parse_args(argv), extra_args


def run_simulation(all_dags, modified_refs):
    """Simulate modifying ``modified_refs`` and show the DAG paths that would submit.

    Forward freshness flood over the whole repo's DAG: any job downstream of a
    modified artifact goes stale and would be submitted to Batch. The modified
    ref can be any type (ds:/fs:/model:) -- staleness floods forward regardless.
    No AWS, no launch.
    """
    global_nodes = [node for nodes in all_dags.values() for node in nodes]
    produced = {ref for node in global_nodes for ref in node.outputs}
    # plan() warns (to stdout) for no-input jobs -- CloudWatch-useful in the
    # Lambda but noise here, and our summary already counts them, so swallow it.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        plan = PipelineManager.from_jobs(global_nodes)._plan(simulated_mtime(modified_refs))
    runs = [(node, reason) for node, should_run, reason in plan if should_run]
    triggered = sum(1 for _, reason in runs if reason in ("stale", "upstream", "missing"))
    always_run = sum(1 for _, reason in runs if reason in ("no_inputs", "unmanaged"))

    print(f"\nSimulating modification of: {', '.join(modified_refs)}")
    print(f"  {triggered} run(s) triggered by this change; {always_run} always-run regardless.\n")

    print("Submitted paths:")
    lines = render_dataflow(
        global_nodes, lambda node: node.mode, produced, roots=list(modified_refs), highlight=modified_refs
    )
    print("\n".join(lines) + "\n")

    # The always-run nodes aren't downstream of the change, so they don't appear
    # in the paths above -- list them so it's clear they submit regardless.
    always = [node for node, reason in runs if reason in ("no_inputs", "unmanaged")]
    if always:
        print("Always-run regardless of this change:")
        for node in always:
            print(f"   {script_label(node.stem, node.mode)}")
        print()


def main():
    args, extra_args = parse_args()

    # Discover all pipelines and DAG definitions
    all_pipelines, all_dags = get_all_pipelines()
    if not all_pipelines:
        print(f"No pipeline scripts found in subdirectories of {Path.cwd()}")
        exit(1)
    if not all_dags:
        log.warning(
            f"No pipelines.json found under {Path.cwd()} -- running scripts standalone "
            f"(no DAG ordering or modes). To run DAG nodes, cd to a directory with the relevant pipelines.json."
        )

    # Freshness simulation: show what a modified source would submit (no launch).
    if args.sim_mod:
        run_simulation(all_dags, args.sim_mod)
        return

    # Select which pipelines to run
    selected, selected_keys, force_keys, selection_desc = select_pipelines(all_pipelines, all_dags, args)

    # Validate --local before doing more work
    if args.local and len(selected) > 1:
        print(f"\n--local only supports a single script, but {len(selected)} matched:")
        for p in selected:
            print(f"   {p.name}")
        print("\nNarrow your selection to a single script.")
        exit(1)

    # Resolve execution mode
    mode = resolve_mode(args)

    # Build execution plan (mtime-aware: only stale jobs run). Hand the manager
    # workbench's region-bound, assumed-role session so it can resolve mtimes.
    from workbench.core.cloud_platform.aws.aws_session import AWSSession

    session = AWSSession().boto3_session
    try:
        plan = sort_pipelines(
            selected, all_dags, mode, selected_keys, full_dag=args.full_dag, session=session, force_keys=force_keys
        )
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

    # Wrong-environment guard: if nearly every artifact came back missing, this is almost
    # certainly the wrong AWS account/region rather than a real from-scratch build. Refuse
    # to submit the whole world unless the user explicitly forces it.
    if plan.environment_suspect and not args.force:
        print(
            "\nABORTED: nearly every artifact was not found -- this usually means the wrong AWS "
            "account/region (check WORKBENCH_CONFIG). Re-run with --force if this is intentional "
            "(e.g. a genuine from-scratch build).\n"
        )
        exit(1)

    run_pipelines(plan, args, extra_args)


if __name__ == "__main__":
    main()
