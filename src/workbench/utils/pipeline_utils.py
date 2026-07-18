"""Helpers for the ML pipeline hierarchy (CachedMeta.pipelines())."""


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
