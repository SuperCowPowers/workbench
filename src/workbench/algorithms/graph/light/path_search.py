"""Path search across a proximity graph."""

import networkx as nx
import logging

# Set up logging
log = logging.getLogger("workbench")


def shortest_path(graph: nx.Graph, source, target) -> list:
    """Find the most-similar chain of nodes connecting source to target.

    Edge `weight` in a proximity graph is a similarity in [0, 1], so traversal cost is
    1 - similarity: the path prefers hops between close neighbors, even if that takes
    more of them.

    Args:
        graph (nx.Graph): A proximity graph (edges carry a `weight` similarity).
        source: The starting node ID.
        target: The destination node ID.

    Returns:
        list: Node IDs from source to target, empty if the two aren't connected.
    """
    for node in (source, target):
        if node not in graph:
            raise ValueError(f"Node ID '{node}' not found in the graph.")

    try:
        return nx.shortest_path(graph, source, target, weight=lambda u, v, d: 1.0 - d.get("weight", 0.0))
    except nx.NetworkXNoPath:
        log.warning(f"No path between '{source}' and '{target}' — they're in different components.")
        return []


def path_subgraph(graph: nx.Graph, path: list, radius: int = 1) -> nx.Graph:
    """Get the subgraph of a path plus the nodes surrounding it.

    Args:
        graph (nx.Graph): The graph the path came from.
        path (list): Node IDs along the path.
        radius (int): Hops of surrounding context to include (default: 1, 0 for the path alone).

    Returns:
        nx.Graph: The induced subgraph, with an `on_path` node attribute flagging path members.
    """
    path_nodes = set(path)
    nodes = set(path_nodes)
    for _ in range(radius):
        nodes.update(neighbor for node in list(nodes) for neighbor in graph.neighbors(node))

    subgraph = graph.subgraph(nodes).copy()
    nx.set_node_attributes(subgraph, {n: int(n in path_nodes) for n in subgraph}, "on_path")
    return subgraph


if __name__ == "__main__":
    import pandas as pd
    from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity

    # A chain of compounds where f1 climbs steadily
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "f1": [0.0, 0.1, 0.2, 0.3, 0.4],
            "f2": [0.0, 0.1, 0.2, 0.3, 0.4],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    graph = FeatureSpaceProximity(df, id_column="id", features=["f1", "f2"], target="target").graph(n_neighbors=2)

    path = shortest_path(graph, "a", "e")
    print(f"Path: {path}")

    context = path_subgraph(graph, path, radius=1)
    print(f"Path subgraph: {context.number_of_nodes()} nodes, {context.number_of_edges()} edges")
    print("On path:", nx.get_node_attributes(context, "on_path"))

    # Disconnected nodes return an empty path
    disconnected = nx.Graph()
    disconnected.add_nodes_from(["x", "y"])
    print(f"Disconnected: {shortest_path(disconnected, 'x', 'y')}")
