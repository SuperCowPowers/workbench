"""pipeline_serializer: shape PipelineManager's DAGs into the dashboard's wire format.

Internal serialization behind ``Meta.pipelines()`` / ``Meta.pipeline()`` -- not a
public class. PipelineManager stays the single source of truth for pipelines.json
semantics (including the relative_dir grouping it discovers); this module only
*shapes* its graphs into a JSON/Redis-safe form (node-link dicts) and nests them into
the arbitrary-depth group tree the UI renders.

A node-link dict is ``{"nodes": [...], "links": [{"source", "target"}]}`` and round-
trips to a live graph via ``networkx.node_link_graph(d, edges="links")``.

The hierarchy is a group tree keyed by each pipeline's relative_dir (the directory
segments above its pipelines.json -- an arbitrary-depth client grouping). A group is
``{"name": str, "subgroups": [group, ...], "pipelines": {name: node_link_dict}}``;
the leaf dir holding a pipelines.json carries the pipelines, its ancestors are nesting.
"""

import logging
from typing import Optional

from workbench.lambda_layer.pipeline_manager import PipelineManager, ref_name, ref_type

log = logging.getLogger("workbench")


def pipeline_hierarchy(root: str, session=None) -> list:
    """Build the full pipeline hierarchy as a group tree of node-link dicts.

    Args:
        root (str): ML_PIPELINES_ROOT -- a local directory or ``s3://`` prefix.
        session: Optional boto3 session for S3 discovery.

    Returns:
        list: Top-level groups, each
            ``{"name", "subgroups": [group, ...], "pipelines": {name: node_link_dict}}``.
    """
    if not root:
        return []

    pm = _load(root, session)
    if pm is None:
        return []

    root_name = _root_name(root)
    root_group = _new_group("")  # synthetic root; its subgroups are the top-level groups
    for name in pm.list_pipelines():
        try:
            subgraph = pm.get_pipeline(name)
        except KeyError:
            continue
        # Walk (creating as needed) the group path for this pipeline's relative_dir,
        # then attach the pipeline to the leaf group.
        node = root_group
        for seg in _group_segments(pm.get_pipeline_relative_dir(name), root_name):
            node = node["subgroups"].setdefault(seg, _new_group(seg))
        if name in node["pipelines"]:
            log.warning(f"Duplicate pipeline name {name!r} under {node['name']!r}; keeping first")
            continue
        node["pipelines"][name] = linearize(subgraph)
    return _finalize(root_group)["subgroups"]


def single_pipeline(root: str, name: str, session=None) -> Optional[dict]:
    """Build one pipeline's graph as a node-link dict (or None if not found)."""
    if not root:
        return None
    pm = _load(root, session)
    if pm is None:
        return None
    try:
        return linearize(pm.get_pipeline(name))
    except KeyError:
        log.warning(f"No pipeline named {name!r} under {root}")
        return None


def promotion_map(root: str, session=None) -> dict:
    """Map each promotion endpoint to its challenger models.

    A promotion job is any pipeline node whose script stem starts with
    ``model_promotion`` (the core arbiter or a client override). Its endpoint
    output is a champion endpoint; its model inputs are the challengers.
    PipelineManager enforces one producer per artifact, so each endpoint maps
    to exactly one promotion job.

    Args:
        root (str): ML_PIPELINES_ROOT -- a local directory or ``s3://`` prefix.
        session: Optional boto3 session for S3 discovery.

    Returns:
        dict: ``{endpoint_name: [challenger model names]}``
    """
    pm = _load(root, session) if root else None
    if pm is None:
        return {}

    promo: dict = {}
    for job in pm.jobs:
        if not job.stem.startswith("model_promotion"):
            continue
        challengers = sorted(ref_name(i) for i in job.inputs if ref_type(i) == "model")
        for out in job.outputs:
            if ref_type(out) == "endpoint":
                promo[ref_name(out)] = challengers
    return promo


def _load(root: str, session) -> Optional[PipelineManager]:
    """Construct a PipelineManager, tolerating a malformed root (log, don't crash)."""
    try:
        return PipelineManager(root, session=session)
    except Exception as e:
        log.error(f"Failed to load pipelines from {root}: {e}")
        return None


def _serialize(subgraph) -> dict:
    """Convert a PipelineManager sub-DAG into a JSON/Redis-safe node-link dict.

    Job nodes get a *globally* unique id: a job's ``node_id`` is a human label that can
    repeat when one script drives several stages, and a per-graph index isn't unique
    either -- the UI merges several pipelines into one card and dedups nodes by id, so
    ids must be unique across pipelines. A job's outputs are globally unique (one
    producer per ref), so they key the job. Artifact nodes keep their ref and type; job
    nodes carry only their id (``linearize`` collapses them, keyed by that id).
    """
    ids = {n: (_job_id(d["job"]) if d.get("kind") == "job" else n) for n, d in subgraph.nodes(data=True)}
    nodes = [
        (
            {"id": ids[n], "kind": "job"}
            if d.get("kind") == "job"
            else {"id": n, "kind": "artifact", "type": d.get("type")}
        )
        for n, d in subgraph.nodes(data=True)
    ]
    links = [{"source": ids[u], "target": ids[v]} for u, v in subgraph.edges()]
    return {"nodes": nodes, "links": links}


def _job_id(job) -> str:
    """A globally-unique node id for a job, keyed by its (globally-unique) outputs.

    Falls back to script+mode for the rare output-less job (still unique -- PM forbids
    two jobs with the same key). The "job:" prefix can't collide with an artifact ref.
    """
    if job.outputs:
        return "job:" + ";".join(sorted(job.outputs))
    return f"job:{job.stem}:{job.mode}"


# Canonical artifact-lineage order. A model always derives from a feature set, an
# endpoint from a model -- a fact of the platform, so threading by it is not inference.
_TYPE_BAND = {"ds": 0, "public": 0, "fs": 1, "model": 2, "endpoint": 3}


def _band(artifact_type) -> int:
    return _TYPE_BAND.get(artifact_type, 2)


def linearize(subgraph) -> dict:
    """The lineage view the UI renders: an artifact-only DAG.

    Collapses each job to artifact -> artifact edges, but instead of fanning the job's
    input to every output, it threads the job's *own* artifacts up the type ladder
    (ds -> fs -> model -> endpoint). So a single script that produces ds -> {fs, model,
    endpoint} renders as the chain, not a fan -- with no cross-job inference and no
    invented artifacts.

    When one job produces several artifacts in *both* of two adjacent bands (e.g. 4
    models + 4 endpoints), it pairs them by ref-name: an endpoint is named after its
    source model (``Model.to_endpoint``), so ``model:x -> endpoint:x`` is real identity,
    not a guess. If the names don't line up 1:1 (a genuinely ambiguous many-to-many),
    the whole job falls back to the plain input->output fan -- a visible "split this
    job" signal rather than a guessed mesh.
    """
    node_link = _serialize(subgraph)
    art_type = {n["id"]: n["type"] for n in node_link["nodes"] if n["kind"] == "artifact"}
    jobs = {n["id"] for n in node_link["nodes"] if n["kind"] == "job"}

    inbound: dict = {}
    outbound: dict = {}
    for link in node_link["links"]:
        if link["target"] in jobs:
            inbound.setdefault(link["target"], []).append(link["source"])
        if link["source"] in jobs:
            outbound.setdefault(link["source"], []).append(link["target"])

    edges = set()
    for job in jobs:
        ins, outs = inbound.get(job, []), outbound.get(job, [])
        by_band: dict = {}
        for ref in ins + outs:
            by_band.setdefault(_band(art_type.get(ref)), []).append(ref)
        threaded = _thread_bands(by_band)
        if threaded is not None:
            edges |= threaded
        else:  # ambiguous or single-band job -> plain input->output fan
            for a in ins:
                for b in outs:
                    if a != b:
                        edges.add((a, b))

    nodes = [{"id": ref, "type": t} for ref, t in art_type.items()]
    links = [{"source": a, "target": b} for a, b in sorted(edges)]
    return {"nodes": nodes, "links": links}


def _thread_bands(by_band: dict) -> Optional[set]:
    """Thread a job's artifacts up the type ladder, one adjacent band pair at a time.

    A pair with a singleton on either side is an unambiguous star. A many-to-many pair
    is paired by ref-name (model:x -> endpoint:x). Returns the edge set, or None if the
    job spans a single band or a many-to-many pair can't be name-paired 1:1 (the caller
    then falls back to a plain fan).
    """
    bands = sorted(by_band)
    if len(bands) <= 1:
        return None
    edges = set()
    for lo, hi in zip(bands, bands[1:]):
        los, his = by_band[lo], by_band[hi]
        if len(los) == 1 or len(his) == 1:
            edges.update((a, b) for a in los for b in his)
        else:
            paired = _name_pairs(los, his)
            if paired is None:
                return None  # ambiguous many-to-many -> whole-job fallback
            edges |= paired
    return edges


def _name_pairs(los: list, his: list) -> Optional[set]:
    """Pair each hi artifact to its same-named lo artifact (model:x -> endpoint:x).

    Returns None unless every hi ref matches exactly one lo ref by name -- an orphan or
    ambiguous name means the pairing isn't real identity. Unmatched lo refs are fine
    (a model with no endpoint just terminates its chain).
    """
    by_name: dict = {}
    for ref in los:
        by_name.setdefault(_ref_name(ref), []).append(ref)
    edges = set()
    for h in his:
        matches = by_name.get(_ref_name(h), [])
        if len(matches) != 1:
            return None
        edges.add((matches[0], h))
    return edges


def _ref_name(ref: str) -> str:
    """The name portion of an artifact ref (``model:x`` -> ``x``)."""
    return ref.split(":", 1)[1]


def _new_group(name: str) -> dict:
    """A group node with its subgroups still keyed by name (a dict) for O(1) walking."""
    return {"name": name, "subgroups": {}, "pipelines": {}}


# Upstream/shared groups that sort before everything else, regardless of name (a
# load_data step feeds the siblings that consume its featuresets).
_SOURCE_FIRST = {"load_data"}
# Catch-all groups that sort after everything else, regardless of name.
_SINK_LAST = {"misc"}


def _finalize(node: dict) -> dict:
    """Emit the wire form: subgroups become a name-sorted list (recursively).

    Sort is alphabetical, except source groups (see _SOURCE_FIRST) float to the top and
    catch-all groups (see _SINK_LAST) sink to the bottom.
    """
    children = sorted(
        node["subgroups"].values(),
        key=lambda g: (
            g["name"].lower() not in _SOURCE_FIRST,
            g["name"].lower() in _SINK_LAST,
            g["name"].lower(),
        ),
    )
    return {
        "name": node["name"],
        "subgroups": [_finalize(child) for child in children],
        "pipelines": node["pipelines"],
    }


def _group_segments(relative_dir: str, root_name: str) -> list:
    """Group path for a pipeline: its relative_dir segments, arbitrary depth.

    A root-level pipelines.json (empty relative_dir) has no directory grouping, so it
    lands under a single top-level group named after the root.
    """
    parts = [p for p in relative_dir.split("/") if p]
    return parts if parts else [root_name]


def _root_name(root: str) -> str:
    """Last path segment of the root (the group name for a root-level pipelines.json)."""
    r = str(root).rstrip("/")
    return r.rsplit("/", 1)[-1] if "/" in r else r
