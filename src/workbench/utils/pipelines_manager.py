"""Graph-first pipeline manager: the single source of truth for pipelines.json.

This module interprets the pipelines.json schema and builds the dependency DAG
that drives both the launcher (laptop -> AWS Batch) and the nightly DT Lambda.
Both import this *same* module (the Lambda gets it from a shared layer), so the
two paths run identical graph logic.

Two layers, deliberately separated:

  * **Artifact DAG** -- the graph of typed artifacts (``ds:``/``fs:``/``model:``
    /``endpoint:``). Each artifact is a node carrying the job that produces it.
    A *root* is any artifact with no in-graph producer -- a DataSource, an
    externally-built FeatureSet, a hand-uploaded Model: the type does not
    matter. This layer answers the topology questions: full DAG, sub-DAG,
    ancestors/descendants of an artifact.

  * **Jobs** -- a job is one script run ``(script, mode)`` with N declared
    inputs and N declared outputs. Jobs are the submission unit: one job = one
    Batch submission, producing all its outputs together.

Freshness flows *forward* over the graph (Dagster-style "stale" propagation): a
job is stale when one of its outputs is missing, an immediate input is newer
than its outputs, or any immediate upstream job is itself (re)running. Processed
in topological order, a single pass pushes a whole ``ds -> fs -> model`` chain
even though the intermediate artifacts have not been rebuilt yet. There is no
backtracking to a root DataSource -- the forward flood subsumes it, and nothing
special-cases artifact type.

Resolving a ref to a modified time needs AWS, so callers inject an
``mtime_fn(ref) -> datetime | None``; this module stays I/O-free and unit-testable
with a fake clock.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import networkx as nx


def ref_type(ref: str) -> str:
    """Type prefix of an artifact ref, e.g. 'fs:caco2_1' -> 'fs'."""
    return ref.partition(":")[0]


def ref_name(ref: str) -> str:
    """Name portion of an artifact ref, e.g. 'fs:caco2_1' -> 'caco2_1'."""
    return ref.partition(":")[2]


@dataclass
class Job:
    """One script run: a ``script`` in an optional ``mode`` with typed refs.

    A job declares the artifact refs it produces (outputs) and depends on
    (inputs). It is the submission unit -- running it regenerates *all* its
    outputs. ``script`` is kept generic (a local ``Path`` for the launcher, an
    S3 URI string in the Lambda) since this module uses it only for identity and
    labeling.

    Attributes:
        script (Any): The script identity (Path locally, S3 URI in the Lambda)
        mode (str | None): Execution mode ("dt"/"ts"/...), or None for modeless
        outputs (list[str]): Artifact refs this job produces
        inputs (list[str]): Artifact refs this job depends on
        group (str | None): The pipeline this job belongs to (its name)
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


def parse_spec(spec: dict, script_resolver: Callable[[str], Any] | None = None) -> list[Job]:
    """Parse one pipelines.json dict into Jobs, tagged by pipeline name.

    To resolve dependencies across multiple pipelines.json files, parse each and
    concatenate the lists before constructing a :class:`PipelineGraph`.

    Args:
        spec (dict): Parsed pipelines.json ({"pipelines": {name: [raw, ...]}})
        script_resolver (callable | None): Maps a raw "script" string to the
            identity this context wants (e.g. an S3 URI or a local Path).
            Identity when omitted.

    Returns:
        list[Job]: One job per declared entry, with ``group`` set.
    """
    jobs: list[Job] = []
    for group, raw_jobs in spec.get("pipelines", {}).items():
        for raw in raw_jobs:
            script = raw["script"]
            if script_resolver is not None:
                script = script_resolver(script)
            jobs.append(
                Job(
                    script=script,
                    mode=raw.get("mode"),
                    outputs=list(raw.get("outputs", [])),
                    inputs=list(raw.get("inputs", [])),
                    group=group,
                )
            )
    return jobs


class PipelineGraph:
    """The dependency DAG over a set of jobs, plus forward-flood scheduling.

    Build it from a flat list of :class:`Job` (concatenate ``parse_spec`` results
    across every pipelines.json to resolve cross-file dependencies). It exposes
    the artifact topology (``ancestors``/``descendants``/``subdag``) and the job
    scheduler (``plan``/``submission_order``).

    Attributes:
        jobs (list[Job]): Every job under consideration.
        graph (nx.DiGraph): The artifact DAG. Nodes are artifact refs; each node
            carries ``producer`` (the Job, or None for a root), ``group`` and
            ``type``. Edges run input-ref -> output-ref for every job.
        job_graph (nx.DiGraph): The job DAG. Nodes are ``job.key`` (carrying the
            Job under "job"); edges run producer-job -> consumer-job.
    """

    def __init__(self, jobs: list[Job]):
        self.jobs = list(jobs)

        # Each (script, mode) is a unique run.
        seen: set = set()
        for job in self.jobs:
            if job.key in seen:
                raise ValueError(f"duplicate job {job.node_id!r} ((script, mode) declared more than once)")
            seen.add(job.key)

        # ref -> the one job that outputs it. Each artifact needs exactly one
        # producer; two producers is a hard schema error.
        self._producer: dict[str, Job] = {}
        for job in self.jobs:
            for ref in job.outputs:
                existing = self._producer.get(ref)
                if existing is not None and existing is not job:
                    raise ValueError(
                        f"artifact {ref!r} is produced by both {existing.node_id!r} "
                        f"and {job.node_id!r} (each artifact needs exactly one producer)"
                    )
                self._producer[ref] = job

        self.graph = self._build_artifact_graph()
        self.job_graph = self._build_job_graph()  # also validates: no cycles

    # -- construction ---------------------------------------------------------

    def _build_artifact_graph(self) -> nx.DiGraph:
        """Artifact DAG: nodes are refs, edges run each input -> each output."""
        g = nx.DiGraph()
        for job in self.jobs:
            for ref in (*job.inputs, *job.outputs):
                if ref not in g:
                    producer = self._producer.get(ref)
                    g.add_node(
                        ref,
                        producer=producer,
                        group=producer.group if producer else None,
                        type=ref_type(ref),
                    )
            for inp in job.inputs:
                for out in job.outputs:
                    g.add_edge(inp, out)
        return g

    def _build_job_graph(self) -> nx.DiGraph:
        """Job DAG: nodes are job.key, edges run producer-job -> consumer-job."""
        g = nx.DiGraph()
        for job in self.jobs:
            g.add_node(job.key, job=job)
        for job in self.jobs:
            for ref in job.inputs:
                producer = self._producer.get(ref)
                if producer is not None and producer is not job:
                    g.add_edge(producer.key, job.key)

        if not nx.is_directed_acyclic_graph(g):
            cycle = nx.find_cycle(g)
            readable = " -> ".join(g.nodes[k]["job"].node_id for k, _ in cycle)
            raise ValueError(f"pipeline dependency cycle: {readable}")
        return g

    # -- artifact topology ----------------------------------------------------

    def producer(self, ref: str) -> Job | None:
        """The job that produces ``ref``, or None if it is a root/unknown."""
        return self._producer.get(ref)

    def roots(self) -> list[str]:
        """Artifacts with no in-graph producer (sources of any type)."""
        return [r for r in self.graph if self.graph.in_degree(r) == 0]

    def ancestors(self, ref: str) -> set[str]:
        """Every artifact that (transitively) feeds ``ref``."""
        return nx.ancestors(self.graph, ref)

    def descendants(self, ref: str) -> set[str]:
        """Every artifact (transitively) downstream of ``ref``.

        This is the "what does a change to ``ref`` affect" query -- and ``ref``
        can be any type, including a FeatureSet or Model used as a root.
        """
        return nx.descendants(self.graph, ref)

    def subdag(self, refs, *, upstream: bool = True, downstream: bool = False) -> nx.DiGraph:
        """The artifact sub-DAG around ``refs``.

        Args:
            refs (iterable[str]): The artifacts of interest.
            upstream (bool): Include everything feeding them (dependencies).
            downstream (bool): Include everything they feed (dependents).

        Returns:
            nx.DiGraph: The induced subgraph (a copy), ``refs`` always included.
        """
        nodes = set(refs)
        for ref in refs:
            if upstream:
                nodes |= self.ancestors(ref)
            if downstream:
                nodes |= self.descendants(ref)
        return self.graph.subgraph(nodes).copy()

    # -- job selection / ordering --------------------------------------------

    def job_order(self, subset: list[Job] | None = None) -> list[Job]:
        """Jobs in topological order (every producer before its consumers).

        Args:
            subset (list[Job] | None): Restrict to these jobs (still globally
                ordered); all jobs when omitted.
        """
        keys = None if subset is None else {j.key for j in subset}
        ordered = []
        for key in nx.topological_sort(self.job_graph):
            if keys is None or key in keys:
                ordered.append(self.job_graph.nodes[key]["job"])
        return ordered

    def select(self, targets: list[Job]) -> list[Job]:
        """Expand ``targets`` to include their transitive upstream producer jobs.

        The dependency sub-DAG (as jobs) for the things you asked to run, so a
        target's prerequisites come along. Order is not meaningful here -- pass
        the result through :meth:`job_order`.
        """
        keys = set()
        for job in targets:
            keys.add(job.key)
            keys |= nx.ancestors(self.job_graph, job.key)
        return [self.job_graph.nodes[k]["job"] for k in keys]

    # -- scheduling (forward staleness flood) --------------------------------

    def _needs_run(self, job: Job, mtime_fn, running: set) -> tuple[bool, str]:
        """Whether ``job`` must run, given which upstream jobs already will.

        Forward flood -- looks only at immediate inputs and whether an immediate
        upstream job is in ``running``; depth is handled by processing jobs in
        topological order. No backtracking to a root.

        Two bias-to-run cases never silently skip:
          - no declared outputs ("unmanaged"): can't reason about freshness.
          - no declared inputs ("no_inputs"): can't assess staleness, so warn and
            regenerate. (The script is expected to self-gate cheaply.)
        """
        if not job.outputs:
            return True, "unmanaged"

        out_times = [mtime_fn(ref) for ref in job.outputs]
        if any(t is None for t in out_times):
            return True, "missing"
        out_time = min(out_times)

        # An immediate upstream job is (re)running -> its outputs will be fresh,
        # so this job is stale regardless of current timestamps. This is the
        # forward flood that pushes a whole chain in one pass.
        for ref in job.inputs:
            producer = self._producer.get(ref)
            if producer is not None and producer is not job and producer.key in running:
                return True, "upstream"

        if not job.inputs:
            print(
                f"WARNING: job {job.node_id!r} declares no inputs; running "
                f"unconditionally (cannot assess freshness -- declare inputs or mark it a root)."
            )
            return True, "no_inputs"

        in_times = [t for ref in job.inputs if (t := mtime_fn(ref)) is not None]
        if in_times and max(in_times) > out_time:
            return True, "stale"
        return False, "up_to_date"

    def plan(self, mtime_fn) -> list[tuple[Job, bool, str]]:
        """Schedule the whole DAG: (job, should_run, reason) in topo order.

        This is *the scheduler*. Walk jobs producer-before-consumer; each job's
        decision can see which upstream jobs are already slated to run, so
        staleness floods forward and the plan contains the full set of paths to
        push. The caller submits the should_run jobs in this order; Batch's
        input/output dependency tokens enforce ordering within each path.

        Reasons: missing / stale / upstream / unmanaged / no_inputs / up_to_date.
        """
        running: set = set()
        decisions: list[tuple[Job, bool, str]] = []
        for key in nx.topological_sort(self.job_graph):
            job = self.job_graph.nodes[key]["job"]
            should_run, reason = self._needs_run(job, mtime_fn, running)
            if should_run:
                running.add(key)
            decisions.append((job, should_run, reason))
        return decisions

    def submission_order(self, mtime_fn) -> list[Job]:
        """The should_run jobs from :meth:`plan`, in submission order."""
        return [job for job, should_run, _ in self.plan(mtime_fn) if should_run]

    def stale_artifacts(self, mtime_fn) -> set[str]:
        """Artifacts that a run would regenerate (outputs of every running job).

        The artifact-level view of :meth:`plan`, e.g. for rendering the sub-DAG
        a change would push.
        """
        return {ref for job, should_run, _ in self.plan(mtime_fn) if should_run for ref in job.outputs}


def simulated_mtime(modified_refs):
    """An ``mtime_fn`` that marks ``modified_refs`` as freshly modified.

    For simulating freshness propagation without AWS: the given refs report a
    newer time, every other artifact an older-but-present time. Feed it to
    :meth:`PipelineGraph.plan` to see which jobs a change to those refs triggers.

    Because freshness now floods forward (no backtracking), simulating *any* ref
    -- a DataSource, an intermediate FeatureSet, even a Model -- propagates to
    its descendants. The job that produces a simulated ref does not re-run (its
    output is, by definition, already fresh); its consumers do.
    """
    modified = set(modified_refs)
    return lambda ref: 1 if ref in modified else 0
