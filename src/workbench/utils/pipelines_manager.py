"""Pipeline DAG manager: the single source of truth for pipelines.json semantics.

This module interprets the pipelines.json node schema and owns everything about
how nodes wire into a dependency graph and how freshness is decided. It is the
*canonical* copy; verbatim copies are vendored into Lambda asset directories
(e.g. lambdas/ml_pipelines_dt/) so the Lambdas can import it without depending on
workbench. Edit it HERE, then re-copy to the vendored locations.

Because it is vendored into the Lambda runtime, this module is **stdlib-only** --
no workbench imports, no third-party packages (not even networkx). Callers that
want a rich graph object (e.g. the launcher's Unicode tree) build it themselves
from `derive_edges()`.

Schema: a pipelines.json maps each pipeline name to a flat list of nodes. A node
runs a `script` in an optional `mode` and declares the typed artifact refs it
`outputs` and `inputs` (e.g. "fs:caco2_er_reg_1", "ds:...", "model:...",
"endpoint:..."). Edges are derived by matching a node's inputs to whichever node
outputs them -- across *all* nodes handed in, so dependencies that cross
pipelines (or even pipelines.json files) resolve to real edges.

Freshness walks the declared graph: a node needs to run when one of its outputs
is missing or older than the latest upstream source feeding its inputs. The walk
is transitive -- it recurses through in-graph producers down to the leaves
(DataSources / external artifacts), so a model's freshness roots at the
DataSource that ultimately feeds it rather than at an intermediate FeatureSet
(whose only timestamp is a CreationTime, not a real update time). Resolving a ref
to a modified time needs AWS, so callers inject an ``mtime_fn(ref) -> datetime |
None``; this module stays I/O-free and unit-testable with a fake clock.
"""

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

# Typed artifact ref prefixes understood by the schema. A ref is "<type>:<name>".
KNOWN_REF_TYPES = ("ds", "fs", "model", "endpoint")


def ref_type(ref: str) -> str:
    """Type prefix of an artifact ref, e.g. 'fs:caco2_1' -> 'fs'."""
    return ref.partition(":")[0]


def ref_name(ref: str) -> str:
    """Name portion of an artifact ref, e.g. 'fs:caco2_1' -> 'caco2_1'."""
    return ref.partition(":")[2]


@dataclass
class PipelineNode:
    """A single node in a pipeline DAG: a script run in an optional mode.

    The node declares the artifact refs it produces (outputs) and depends on
    (inputs); edges are derived by matching inputs to outputs. ``script`` is kept
    deliberately generic -- the launcher stores a local ``Path``, the Lambda an
    S3 URI string -- since this module only uses it for identity and labeling.

    Attributes:
        script (Any): The script identity (Path locally, S3 URI in the Lambda)
        mode (str | None): Execution mode ("dt"/"ts"), or None for a modeless node
        outputs (list[str]): Artifact refs this node produces
        inputs (list[str]): Artifact refs this node depends on
        group (str | None): The pipeline this node belongs to (its name)
    """

    script: Any
    mode: str | None = None
    outputs: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    group: str | None = None

    @property
    def key(self) -> tuple[Any, str | None]:
        """Identity of this run: (script, mode). Unique across the loaded set."""
        return (self.script, self.mode)

    @property
    def stem(self) -> str:
        """Script filename without directory or .py suffix."""
        name = os.path.basename(str(self.script))
        return name[:-3] if name.endswith(".py") else name

    @property
    def node_id(self) -> str:
        """Human-readable label for messages: 'stem [mode]' or 'stem'."""
        return f"{self.stem} [{self.mode}]" if self.mode else self.stem


def parse_spec(spec: dict, script_resolver: Callable[[str], Any] | None = None) -> list[PipelineNode]:
    """Parse one pipelines.json dict into PipelineNodes, tagged by pipeline name.

    To resolve dependencies across multiple pipelines.json files, parse each and
    concatenate the lists before building the index / ordering.

    Args:
        spec (dict): Parsed pipelines.json ({"pipelines": {name: [raw_node, ...]}})
        script_resolver (callable | None): Maps a raw "script" string to the
            identity this context wants (e.g. an S3 URI or a local Path). Identity
            when omitted.

    Returns:
        list[PipelineNode]: One node per declared entry, with ``group`` set.
    """
    nodes: list[PipelineNode] = []
    for group, raw_nodes in spec.get("pipelines", {}).items():
        for raw in raw_nodes:
            script = raw["script"]
            if script_resolver is not None:
                script = script_resolver(script)
            nodes.append(
                PipelineNode(
                    script=script,
                    mode=raw.get("mode"),
                    outputs=list(raw.get("outputs", [])),
                    inputs=list(raw.get("inputs", [])),
                    group=group,
                )
            )
    return nodes


def build_producer_index(nodes: list[PipelineNode]) -> dict[str, PipelineNode]:
    """Map each output artifact ref to the node that produces it.

    Spans every node handed in, so the index resolves cross-pipeline (and
    cross-file) dependencies. An artifact declared by two nodes is a hard schema
    error -- each artifact needs exactly one producer.
    """
    index: dict[str, PipelineNode] = {}
    for node in nodes:
        for ref in node.outputs:
            existing = index.get(ref)
            if existing is not None and existing is not node:
                raise ValueError(
                    f"artifact {ref!r} is produced by both {existing.node_id!r} "
                    f"and {node.node_id!r} (each artifact needs exactly one producer)"
                )
            index[ref] = node
    return index


def derive_edges(
    nodes: list[PipelineNode],
    index: dict[str, PipelineNode],
) -> tuple[list[tuple[PipelineNode, PipelineNode]], list[tuple[PipelineNode, str]]]:
    """Derive (producer -> consumer) edges and list inputs with no producer.

    Args:
        nodes (list[PipelineNode]): All nodes under consideration
        index (dict): Producer index from build_producer_index()

    Returns:
        tuple: (edges, externals)
            - edges: (producer_node, consumer_node) pairs, one per resolved input
            - externals: (consumer_node, ref) pairs for inputs no node produces
              (DataSources, or artifacts produced outside the selected set)
    """
    edges: list[tuple[PipelineNode, PipelineNode]] = []
    externals: list[tuple[PipelineNode, str]] = []
    for node in nodes:
        for ref in node.inputs:
            producer = index.get(ref)
            if producer is None:
                externals.append((node, ref))
            elif producer is not node:
                edges.append((producer, node))
    return edges, externals


def topo_order(nodes: list[PipelineNode]) -> list[PipelineNode]:
    """Order nodes so every producer precedes its consumers (Kahn's algorithm).

    Spans all nodes, so a producer in one pipeline is ordered before a consumer
    in another. Inputs with no in-set producer add no constraint. A dependency
    cycle is a hard error.
    """
    index = build_producer_index(nodes)  # also validates one-producer-per-artifact
    pos = {id(node): i for i, node in enumerate(nodes)}
    indegree = [0] * len(nodes)
    consumers: list[list[int]] = [[] for _ in nodes]

    for i, node in enumerate(nodes):
        for ref in node.inputs:
            producer = index.get(ref)
            if producer is not None and producer is not node:
                consumers[pos[id(producer)]].append(i)
                indegree[i] += 1

    queue = deque(i for i in range(len(nodes)) if indegree[i] == 0)
    order: list[int] = []
    while queue:
        i = queue.popleft()
        order.append(i)
        for j in consumers[i]:
            indegree[j] -= 1
            if indegree[j] == 0:
                queue.append(j)

    if len(order) < len(nodes):
        seen = set(order)
        remaining = [nodes[i].node_id for i in range(len(nodes)) if i not in seen]
        raise ValueError(f"pipeline dependency cycle involving: {', '.join(remaining)}")
    return [nodes[i] for i in order]


def effective_source_time(ref, index, mtime_fn, _stack=None):
    """Latest upstream source time feeding ``ref``, walking the declared graph.

    If ``ref`` has an in-graph producer, recurse into that producer's inputs and
    return the max (bottoming out at DataSources / external artifacts). If it has
    no producer (a leaf) -- or the producer declares no inputs -- use the ref's
    own modified time. Cycle-guarded via ``_stack``. Returns None when nothing
    upstream resolves to a time (e.g. a missing leaf).
    """
    if _stack is None:
        _stack = frozenset()

    producer = index.get(ref)
    if producer is None or ref in _stack:
        return mtime_fn(ref)

    inputs = producer.inputs
    if not inputs:
        # Producer with no declared inputs: its own output is the signal.
        return mtime_fn(ref)

    times = []
    for inp in inputs:
        t = effective_source_time(inp, index, mtime_fn, _stack | {ref})
        if t is not None:
            times.append(t)
    return max(times) if times else None


def needs_run(node, index, mtime_fn) -> tuple[bool, str]:
    """Decide whether a node must run, returning (should_run, reason).

    A node runs when any declared output is missing, or when the latest upstream
    source feeding its inputs is newer than its oldest output.

    Two bias-to-run cases never silently skip:
      - no declared outputs ("unmanaged"): can't reason about freshness.
      - no declared inputs ("no_inputs"): can't assess staleness, so we warn and
        regenerate rather than risk a stale artifact going unnoticed. (The script
        is expected to self-gate cheaply.)
    """
    outputs = node.outputs
    if not outputs:
        return True, "unmanaged"  # no output to check freshness against

    out_times = [mtime_fn(ref) for ref in outputs]
    if any(t is None for t in out_times):
        return True, "missing"
    out_time = min(out_times)

    inputs = node.inputs
    if not inputs:
        print(
            f"WARNING: node {node.node_id!r} declares no inputs; running "
            f"unconditionally (cannot assess freshness -- declare inputs or mark it a root)."
        )
        return True, "no_inputs"

    in_times = [effective_source_time(ref, index, mtime_fn) for ref in inputs]
    in_times = [t for t in in_times if t is not None]
    if in_times and max(in_times) > out_time:
        return True, "stale"
    return False, "up_to_date"


def plan_runs(nodes, mtime_fn):
    """Schedule the whole global DAG: decide what to run, in dependency order.

    This is *the scheduler*. Treat ``nodes`` as one general DAG (across pipelines
    and files); given a clock, return every node's run decision in topological
    order so a producer is always considered before its consumers.

    Freshness is source-rooted (see needs_run): when a DataSource changes, every
    node on a path downstream of it goes stale, so the returned plan naturally
    contains the *full set of paths that need pushing*. The caller submits the
    should_run nodes to AWS Batch in this order; Batch's input/output dependency
    tokens then enforce ordering within each path.

    Args:
        nodes (list[PipelineNode]): Every node in the global DAG.
        mtime_fn (callable): Resolves an artifact ref to its modified time (or
            None if absent).

    Returns:
        list[tuple[PipelineNode, bool, str]]: (node, should_run, reason) in
            topological order. Reasons: missing / stale / unmanaged / no_inputs /
            up_to_date.
    """
    index = build_producer_index(nodes)
    return [(node, *needs_run(node, index, mtime_fn)) for node in topo_order(nodes)]
