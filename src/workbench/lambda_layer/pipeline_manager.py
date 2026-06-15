"""PipelineManager: the single source of truth for pipelines.json semantics.

This module interprets the pipelines.json schema and builds the dependency DAG
that drives both the launcher (laptop -> AWS Batch) and the nightly DT Lambda.
Both import this *same* module (the Lambda gets it from the workbench layer), so
the two paths run identical graph logic.

One **bipartite** dependency DAG is the single source of truth, modeled the way
Dagster/Airflow/dbt do it -- two node kinds, semantic edges:

  * **Artifact nodes** -- typed artifacts (``ds:``/``fs:``/``model:``/``public:``
    /``endpoint:``). A *root* is any artifact with no producing job: a DataSource,
    an externally-built FeatureSet, a hand-uploaded Model, a ``public:`` dataset --
    the type does not matter.
  * **Job nodes** -- one script run ``(script, mode)`` with N declared inputs and
    N declared outputs. A job is the submission unit (one job = one Batch run).

Edges are directional and semantic: ``artifact -> job`` (the job consumes it) and
``job -> artifact`` (the job produces it). Because there is never a direct
artifact-to-artifact edge, a chain like ``ds -> fs -> model -> endpoint`` is
*structural* -- it can only exist through the producing jobs, so impossible
shortcuts (e.g. a FeatureSet wired straight to an endpoint) are unrepresentable.

The human-facing queries (full graph / upstream / downstream / per-pipeline)
return slices of this one graph -- artifacts *and* the jobs between them -- as
``networkx.DiGraph`` objects you can hand straight to :meth:`PipelineManager.show`.
Scheduling filters the same graph to its job nodes (topological order).

Freshness flows *forward* over the graph (Dagster-style "stale" propagation): a
job is stale when one of its outputs is missing, an immediate input is newer
than its outputs, or any immediate upstream job is itself (re)running. Processed
in topological order, a single pass pushes a whole ``ds -> fs -> model`` chain
even though the intermediate artifacts have not been rebuilt yet -- no
backtracking, nothing special-cases artifact type.

Resolving a ref to a modified time needs AWS. By default the manager does this
itself (raw boto3, cached per instance) -- the normal path for both the launcher
and the Lambda. A clock can be injected (``mtime_fn(ref) -> datetime | None``)
only to simulate (``simulated_mtime``) or to unit-test with a fake clock; that
override never touches the default path.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NamedTuple

import networkx as nx

# boto3 error codes that mean "this artifact does not exist (yet)" -- a legitimate
# "must run". Distinct from auth/region/throttle failures, which mean "could not
# determine freshness" and must NOT be silently treated as must-run.
_ARTIFACT_NOT_FOUND_CODES = {
    "EntityNotFoundException",  # Glue get_table
    "ResourceNotFound",  # SageMaker describe_feature_group
    "ResourceNotFoundException",
    "ValidationException",  # SageMaker list_model_packages against a missing group
}

# If nearly every resolved artifact is absent, it's almost always the wrong AWS
# account/region rather than a genuine from-scratch build.
SUSPICIOUS_MISS_FRACTION = 0.9

# Public datasets live in this anonymous, read-only S3 bucket. Mirrors
# workbench.api.PublicData.BUCKET, duplicated here (rather than imported) to keep the
# layer dependency-light -- importing PublicData would pull pandas. Resolved via an
# unsigned client, so a `public:` ref's mtime needs no credentials.
PUBLIC_DATA_BUCKET = "workbench-public-data"
PUBLIC_DATA_EXTENSIONS = (".parquet", ".csv")


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

    def pipeline_meta(self, serverless: bool = True) -> dict:
        """The PIPELINE_META a run hands its script: mode + the names to create.

        ``model_name``/``endpoint_name`` come from the declared ``model:`` output
        (the DAG is the source of truth). Modeless producers (e.g. FeatureSets)
        get just ``mode``/``serverless`` and no model. Shared by the launcher and
        the DT Lambda; launcher-only modes (promote, filename fallback) wrap this.
        """
        meta: dict = {"serverless": serverless}
        if self.mode is not None:
            meta["mode"] = self.mode
        model_ref = next((o for o in self.outputs if ref_type(o) == "model"), None)
        if model_ref is not None:
            name = ref_name(model_ref)
            meta["model_name"] = name
            meta["endpoint_name"] = name
        return meta


def parse_spec(spec: dict, script_resolver: Callable[[str], Any] | None = None) -> list[Job]:
    """Parse one pipelines.json dict into Jobs, tagged by pipeline name.

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


class PlanItem(NamedTuple):
    """One scheduling decision from :meth:`PipelineManager.plan`.

    Unpacks as ``(job, run, reason)`` for convenience, or use the attributes.

    Attributes:
        job (Job): The job this decision is about.
        run (bool): Whether it should run.
        reason (str): Why -- missing / stale / upstream / unmanaged / no_inputs /
            up_to_date / selected.
    """

    job: Job
    run: bool
    reason: str


class PipelineManager:
    """Loads pipelines.json into a dependency DAG and answers questions about it.

    Construct it with a path -- a local directory or an ``s3://`` URI -- and it
    discovers every ``pipelines.json`` underneath, parses them into one global
    job set (so dependencies that cross files resolve), and builds the graph.
    Use :meth:`from_jobs` to build from in-memory jobs (tests / programmatic).

    Interactive (human) API:
        list_pipelines / get_num_pipelines / get_pipeline   -- named pipelines
        full_dependency_graph / upstream_graph / downstream_graph -- artifact DAG
        show(graph)                                          -- ascii render

    Orchestration API (shared by the launcher + DT Lambda):
        plan(mtime_fn=None, force=None) -> list[PlanItem]    -- what runs, and why
        suspect_environment                                  -- wrong-account guard
    """

    def __init__(self, path: str | Path, session=None):
        # session: an optional boto3 Session for mtime/S3 access. The launcher
        # passes workbench's region-bound, assumed-role session; the Lambda passes
        # None and gets the default client (region from AWS_REGION). Kept as a
        # plain boto3 Session so the manager needs no workbench import.
        self._session = session
        self._init_from_jobs(self._discover(str(path)))

    @classmethod
    def from_jobs(cls, jobs: list[Job], session=None) -> "PipelineManager":
        """Build from an in-memory job list, bypassing pipelines.json discovery."""
        obj = cls.__new__(cls)
        obj._session = session
        obj._init_from_jobs(list(jobs))
        return obj

    # -- discovery ------------------------------------------------------------

    def _discover(self, path: str) -> list[Job]:
        """Find and parse every pipelines.json under ``path`` (local or S3)."""
        return self._discover_s3(path) if path.startswith("s3://") else self._discover_local(path)

    @staticmethod
    def _discover_local(path: str) -> list[Job]:
        root = Path(path)
        jobs: list[Job] = []
        for cfg in sorted(root.rglob("pipelines.json")):
            spec = json.loads(cfg.read_text())
            d = cfg.parent
            jobs += parse_spec(spec, script_resolver=lambda s, d=d: d / s)
        return jobs

    def _discover_s3(self, path: str) -> list[Job]:
        import boto3  # lazy: only the Lambda path needs it (runtime-provided)

        bucket, _, prefix = path[len("s3://") :].partition("/")
        s3 = (self._session or boto3).client("s3")
        jobs: list[Job] = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("pipelines.json"):
                    continue
                spec = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
                d = key.rsplit("/", 1)[0]
                jobs += parse_spec(spec, script_resolver=lambda s, d=d: f"s3://{bucket}/{d}/{s}")
        return jobs

    # -- construction ---------------------------------------------------------

    def _init_from_jobs(self, jobs: list[Job]) -> None:
        self.jobs = jobs
        self.log = logging.getLogger("workbench")  # color + CloudWatch via logging_setup()
        self._mtime_cache: dict = {}  # ref -> mtime, resolved at most once per instance
        self._aws: dict = {}  # boto3 client cache (lazy)

        # Each (script, mode) is a unique run.
        seen: set = set()
        for job in self.jobs:
            if job.key in seen:
                raise ValueError(f"duplicate job {job.node_id!r} ((script, mode) declared more than once)")
            seen.add(job.key)

        # ref -> the one job that outputs it; two producers is a schema error.
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

        self.graph = self._build_graph()  # also validates: no cycles

    def _build_graph(self) -> nx.DiGraph:
        """The bipartite dependency DAG (the single source of truth).

        Artifact nodes (keyed by ref) and job nodes (keyed by ``job.key``), with
        semantic edges: ``artifact -> job`` (consumes) and ``job -> artifact``
        (produces). No artifact-to-artifact edges -- a chain only exists through
        the producing jobs.
        """
        g = nx.DiGraph()
        # Artifact nodes: every ref any job consumes or produces.
        for ref in {r for job in self.jobs for r in (*job.inputs, *job.outputs)}:
            producer = self._producer.get(ref)
            g.add_node(ref, kind="artifact", type=ref_type(ref), group=producer.group if producer else None)
        # Job nodes + their input/output edges.
        for job in self.jobs:
            g.add_node(job.key, kind="job", job=job, group=job.group)
            for inp in job.inputs:
                g.add_edge(inp, job.key)
            for out in job.outputs:
                g.add_edge(job.key, out)

        if not nx.is_directed_acyclic_graph(g):
            cycle = nx.find_cycle(g)
            readable = " -> ".join(self._node_label(g, n) for n, _ in cycle)
            raise ValueError(f"pipeline dependency cycle: {readable}")
        return g

    @staticmethod
    def _node_label(graph: nx.DiGraph, n) -> str:
        """Display label for a node: a job's 'stem [mode]', or the artifact ref."""
        data = graph.nodes[n]
        return data["job"].node_id if data.get("kind") == "job" else str(n)

    # -- pipelines (named, human units) --------------------------------------

    def list_pipelines(self) -> list[str]:
        """The pipeline names, in first-seen order."""
        names: dict[str, None] = {}
        for job in self.jobs:
            if job.group is not None:
                names.setdefault(job.group, None)
        return list(names)

    def get_num_pipelines(self) -> int:
        """How many named pipelines are loaded."""
        return len(self.list_pipelines())

    def get_pipeline(self, name: str) -> nx.DiGraph:
        """The sub-DAG for one pipeline: its jobs plus the artifacts they touch.

        Inputs produced by *other* pipelines appear as roots here (their producing
        job is outside this slice). Raises KeyError for an unknown name.
        """
        group_jobs = [job for job in self.jobs if job.group == name]
        if not group_jobs:
            raise KeyError(f"no pipeline named {name!r}")
        nodes = {job.key for job in group_jobs}
        nodes |= {ref for job in group_jobs for ref in (*job.inputs, *job.outputs)}
        return self.graph.subgraph(nodes).copy()

    def validate_pipeline(self, name: str):
        """FUTURE: cross-check declared wiring against real artifact lineage."""
        raise NotImplementedError("validate_pipeline is not implemented yet")

    # -- dependency graph (artifact-oriented) --------------------------------

    def full_dependency_graph(self) -> nx.DiGraph:
        """The whole artifact DAG, across all pipelines."""
        return self.graph

    def upstream_graph(self, artifact: str) -> nx.DiGraph:
        """The sub-DAG of everything ``artifact`` depends on (plus itself)."""
        nodes = {artifact} | nx.ancestors(self.graph, artifact)
        return self.graph.subgraph(nodes).copy()

    def downstream_graph(self, artifact: str) -> nx.DiGraph:
        """The sub-DAG of everything changing ``artifact`` would impact (plus itself)."""
        nodes = {artifact} | nx.descendants(self.graph, artifact)
        return self.graph.subgraph(nodes).copy()

    @staticmethod
    def show(graph: nx.DiGraph) -> None:
        """Print a bipartite dependency DAG (from the getters above) as a tree.

        Edges flow input-artifact -> job -> output-artifact. Artifact nodes show
        their ref (``ds:``/``fs:``/``model:``); job nodes show the script as
        ``stem [mode]``. A job that produces no artifact still renders -- it's just
        a leaf, no special-casing.
        """
        from workbench.utils.tree_render import render_forest

        children = {n: list(graph.successors(n)) for n in graph}
        labels = {n: PipelineManager._node_label(graph, n) for n in graph}
        roots = sorted((n for n in graph if graph.in_degree(n) == 0), key=str)
        print("\n".join(render_forest(roots, children, labels)))

    # -- orchestration (internal; launcher + DT Lambda) ----------------------

    def _select(self, targets: list[Job]) -> list[Job]:
        """``targets`` plus their transitive upstream producer jobs."""
        keys = set()
        for job in targets:
            keys.add(job.key)
            keys |= {n for n in nx.ancestors(self.graph, job.key) if self.graph.nodes[n].get("kind") == "job"}
        return [self.graph.nodes[k]["job"] for k in keys]

    def _job_order(self) -> list[Job]:
        """All jobs, topologically (every producer before its consumers).

        Topo order over the bipartite graph interleaves artifacts and jobs; the
        job nodes still come out producer-before-consumer (their shared artifact
        sits between them), so filtering to jobs preserves a valid run order.
        """
        return [
            self.graph.nodes[n]["job"]
            for n in nx.topological_sort(self.graph)
            if self.graph.nodes[n].get("kind") == "job"
        ]

    def _needs_run(self, job: Job, mtime_fn, running: set) -> tuple[bool, str]:
        """Whether ``job`` must run, given which upstream jobs already will.

        Forward flood -- looks only at immediate inputs and whether an immediate
        upstream job is in ``running``; depth is handled by processing jobs in
        topological order. No backtracking to a root.
        """
        if not job.outputs:
            return True, "unmanaged"  # nothing to check freshness against

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
            self.log.warning(
                f"job {job.node_id!r} declares no inputs; running unconditionally "
                f"(cannot assess freshness -- declare inputs or mark it a root)."
            )
            return True, "no_inputs"

        in_times = [t for ref in job.inputs if (t := mtime_fn(ref)) is not None]
        if in_times and max(in_times) > out_time:
            return True, "stale"
        return False, "up_to_date"

    def _aws_client(self, name: str):
        import boto3  # lazy: from the Lambda runtime / workbench's boto3

        if name not in self._aws:
            self._aws[name] = (self._session or boto3).client(name)
        return self._aws[name]

    def _public_s3(self):
        """Anonymous (unsigned) S3 client for the public data bucket -- no creds, us-west-2."""
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        if "__public__" not in self._aws:
            self._aws["__public__"] = boto3.client(
                "s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED)
            )
        return self._aws["__public__"]

    def _public_mtime(self, name: str):
        """LastModified of a PublicData object, trying each known extension.

        Returns None if the dataset isn't found under any extension (-> "must run").
        A non-404 error (throttle, etc.) propagates to the caller's ClientError
        handler -- same "don't guess on failure" rule as the other resolvers.
        """
        from botocore.exceptions import ClientError

        s3 = self._public_s3()
        for ext in PUBLIC_DATA_EXTENSIONS:
            try:
                return s3.head_object(Bucket=PUBLIC_DATA_BUCKET, Key=name + ext)["LastModified"]
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                    continue  # try the next extension
                raise
        self.log.warning(f"_artifact_mtime('public:{name}') -> absent (no .parquet/.csv in {PUBLIC_DATA_BUCKET})")
        return None

    def _artifact_mtime(self, ref: str):
        """Resolve a typed artifact ref to its last-modified time (None if absent).

        Raw boto3 so it stays in the layer's dependency budget:
            ds:<name>     -> Glue table UpdateTime
            fs:<name>     -> FeatureGroup CreationTime
            model:<name>  -> latest model package CreationTime
            public:<name> -> PublicData S3 object LastModified (unsigned, no creds)
            endpoint:...  -> not supported yet

        A genuinely-absent artifact returns None -> "must run". A lookup that we
        *couldn't complete* (bad creds, no region, AccessDenied, throttling) is a
        different beast: returning None there would silently resubmit every job, so
        we let it raise instead. NoCredentials/NoRegion/connection errors aren't
        ClientErrors, so they propagate uncaught -- same intent.
        """
        from botocore.exceptions import ClientError

        kind, _, name = ref.partition(":")
        if not name:
            self.log.error(f"Unrecognized artifact ref (no type prefix): {ref!r}")
            return None
        try:
            if kind == "ds":
                return self._aws_client("glue").get_table(DatabaseName="workbench", Name=name)["Table"]["UpdateTime"]
            if kind == "fs":
                return self._aws_client("sagemaker").describe_feature_group(FeatureGroupName=name)["CreationTime"]
            if kind == "model":
                packages = self._aws_client("sagemaker").list_model_packages(
                    ModelPackageGroupName=name, SortBy="CreationTime", SortOrder="Descending", MaxResults=1
                )["ModelPackageSummaryList"]
                return packages[0]["CreationTime"] if packages else None
            if kind == "public":
                return self._public_mtime(name)
            if kind == "endpoint":
                self.log.warning(f"endpoint artifact refs are not supported yet: {ref!r}")
                return None
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in _ARTIFACT_NOT_FOUND_CODES:
                # Artifact doesn't exist -> "must run". Expected on a first build; a *wall*
                # of these is caught by the plan-level guard in plan().
                self.log.warning(f"_artifact_mtime({ref!r}) -> absent ({code})")
                return None
            # Couldn't determine freshness (AccessDenied, throttling, ...). Fail loudly
            # rather than guess "must run" and resubmit everything.
            self.log.error(f"_artifact_mtime({ref!r}) -> lookup failed ({code}); cannot assess freshness")
            raise
        self.log.error(f"Unknown artifact type in ref: {ref!r}")
        return None

    def _cached_mtime(self, ref: str):
        """``_artifact_mtime`` memoized per instance (the freshness walk hits refs many times)."""
        if ref not in self._mtime_cache:
            self._mtime_cache[ref] = self._artifact_mtime(ref)
        return self._mtime_cache[ref]

    @property
    def suspect_environment(self) -> bool:
        """True when almost every resolved artifact was absent -- the classic
        wrong-account/region fingerprint. Only meaningful after a real ``plan()``
        has populated the mtime cache; False if nothing was resolved.
        """
        cache = self._mtime_cache
        if not cache:
            return False
        absent = sum(1 for v in cache.values() if v is None)
        return absent / len(cache) >= SUSPICIOUS_MISS_FRACTION

    def plan(self, mtime_fn=None, force=None) -> list[PlanItem]:
        """Schedule the whole DAG: a :class:`PlanItem` per job in topo order.

        ``mtime_fn`` defaults to the built-in (real, cached) AWS resolver -- the
        normal path. Pass a clock only to simulate (``simulated_mtime``) or test.

        ``force`` is a set of job keys that run unconditionally (reason
        ``selected``), regardless of freshness -- the launcher uses it so a
        pattern-matched script the user just edited runs even when its inputs are
        up to date. A forced job joins the running set, so the forward flood still
        propagates to its downstream consumers. The DT Lambda passes no force.
        """
        force = force or set()
        real = mtime_fn is None  # only the real AWS resolver feeds the wrong-env guard
        if mtime_fn is None:
            mtime_fn = self._cached_mtime
        running: set = set()
        decisions: list[PlanItem] = []
        for job in self._job_order():
            if job.key in force:
                run, reason = True, "selected"
            else:
                run, reason = self._needs_run(job, mtime_fn, running)
            if run:
                running.add(job.key)
            decisions.append(PlanItem(job, run, reason))

        if real and self.suspect_environment:
            absent = sum(1 for v in self._mtime_cache.values() if v is None)
            total = len(self._mtime_cache)
            self.log.warning(
                f"{absent}/{total} artifacts could not be found -- this usually means the wrong "
                f"AWS account/region (check WORKBENCH_CONFIG). Every job will be (re)built."
            )
        return decisions


def simulated_mtime(modified_refs):
    """An ``mtime_fn`` that marks ``modified_refs`` as freshly modified.

    For simulating freshness propagation without AWS: the given refs report a
    newer time, every other artifact an older-but-present time. Feed it to
    :meth:`PipelineManager.plan` to see which jobs a change to those refs
    triggers. Because freshness floods forward (no backtracking), simulating
    *any* ref -- ds:, an intermediate fs:, even a model: -- propagates to its
    descendants; the job that produces a simulated ref does not re-run.
    """
    modified = set(modified_refs)
    return lambda ref: 1 if ref in modified else 0
