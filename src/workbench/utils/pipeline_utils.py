"""Helpers for the ML pipeline hierarchy (CachedMeta.pipelines())."""

import re

# Promotion copies the winning model to "<base-name>-YYMMDD" and gives that copy
# the endpoint. The pipeline declares the base name, so the date must come off
# before looking a promoted model up.
PROMOTION_SUFFIX = re.compile(r"-\d{6}$")


def base_model_name(name: str) -> str:
    """Strip a promotion date suffix, if present.

    Args:
        name (str): A model name, possibly promoted (e.g. "my-model-260715").

    Returns:
        str: The base name the pipeline declares (e.g. "my-model").
    """
    return PROMOTION_SUFFIX.sub("", name)


def find_pipelines(name: str, artifact_type: str = "model", pipelines: list = None) -> list:
    """Find the pipelines that declare an artifact.

    Handles the two things that make this easy to get wrong: the hierarchy nests
    (groups contain subgroups), and promoted models carry a date suffix that the
    pipeline definition does not.

    Args:
        name (str): Artifact name, with or without a promotion date suffix.
        artifact_type (str): One of ds, fs, model, endpoint, public (default: model).
        pipelines (list): Hierarchy to search; defaults to CachedMeta().pipelines().

    Returns:
        list: One {"group", "pipeline", "matched"} dict per hit, empty if none.
    """
    if pipelines is None:
        from workbench.cached.cached_meta import CachedMeta

        pipelines = CachedMeta().pipelines()

    # Try the name as given, then the de-promoted base name
    candidates = [name]
    base = base_model_name(name)
    if base != name:
        candidates.append(base)
    wanted = [f"{artifact_type}:{candidate}" for candidate in candidates]

    hits = []
    for group in pipelines:
        for pipeline_name, pipeline in group["pipelines"].items():
            node_ids = {node["id"] for node in pipeline["nodes"]}
            for node_id in wanted:
                if node_id in node_ids:
                    hits.append({"group": group["name"], "pipeline": pipeline_name, "matched": node_id})
                    break
        hits += find_pipelines(name, artifact_type, group["subgroups"])
    return hits


def endpoint_group_paths(pipelines: list) -> dict:
    """Map each endpoint to its pipeline-hierarchy group path.

    Walks the ML pipeline hierarchy (the CachedMeta.pipelines() structure: nested
    {name, pipelines, subgroups} groups) and records, for every endpoint node, the group
    names from the root down to it. This is the grouping the ML Pipelines and Contests
    pages display.

    Args:
        pipelines (list): The pipeline hierarchy from CachedMeta().pipelines().

    Returns:
        dict: {endpoint_name: [group path]}. Endpoints not in any pipeline are absent.
    """
    groups = {}

    def walk(nodes, path):
        for g in nodes:
            p = path + [g["name"]]
            for graph in (g.get("pipelines") or {}).values():
                for node in graph.get("nodes", []):
                    if node.get("type") == "endpoint":
                        groups.setdefault(node["id"].split(":", 1)[-1], p)
            walk(g.get("subgroups") or [], p)

    walk(pipelines or [], [])
    return groups
