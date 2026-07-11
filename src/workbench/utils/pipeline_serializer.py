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

from workbench.lambda_layer.pipeline_manager import PipelineManager

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
        node["pipelines"][name] = _serialize(subgraph)
    return _finalize(root_group)["subgroups"]


def single_pipeline(root: str, name: str, session=None) -> Optional[dict]:
    """Build one pipeline's graph as a node-link dict (or None if not found)."""
    if not root:
        return None
    pm = _load(root, session)
    if pm is None:
        return None
    try:
        return _serialize(pm.get_pipeline(name))
    except KeyError:
        log.warning(f"No pipeline named {name!r} under {root}")
        return None


def _load(root: str, session) -> Optional[PipelineManager]:
    """Construct a PipelineManager, tolerating a malformed root (log, don't crash)."""
    try:
        return PipelineManager(root, session=session)
    except Exception as e:
        log.error(f"Failed to load pipelines from {root}: {e}")
        return None


def _serialize(subgraph) -> dict:
    """Convert a PipelineManager sub-DAG into a JSON/Redis-safe node-link dict.

    Job nodes (keyed internally by a tuple, holding a live Job object) are re-keyed to
    their string node_id and flattened to {script, mode}; artifact nodes keep their ref
    and type.
    """
    nodes = []
    for n, d in subgraph.nodes(data=True):
        if d.get("kind") == "job":
            job = d["job"]
            nodes.append({"id": job.node_id, "kind": "job", "script": _script_name(job.script), "mode": job.mode})
        else:
            nodes.append({"id": n, "kind": "artifact", "type": d.get("type")})
    links = [{"source": _node_id(subgraph, u), "target": _node_id(subgraph, v)} for u, v in subgraph.edges()]
    return {"nodes": nodes, "links": links}


def _node_id(subgraph, n):
    """Serializable id for a node: a job's string node_id, else the artifact ref."""
    d = subgraph.nodes[n]
    return d["job"].node_id if d.get("kind") == "job" else n


def _script_name(script) -> str:
    """Basename of a job script (strips directories and any scheme prefix)."""
    return str(script).rsplit("/", 1)[-1]


def _new_group(name: str) -> dict:
    """A group node with its subgroups still keyed by name (a dict) for O(1) walking."""
    return {"name": name, "subgroups": {}, "pipelines": {}}


def _finalize(node: dict) -> dict:
    """Emit the wire form: subgroups become an ordered list (recursively)."""
    return {
        "name": node["name"],
        "subgroups": [_finalize(child) for child in node["subgroups"].values()],
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
