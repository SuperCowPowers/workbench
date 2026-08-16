"""Build a NetworkX graph from a Proximity instance."""

import networkx as nx
import pandas as pd
import logging

# Workbench Imports
from workbench.algorithms.dataframe.proximity import Proximity

# Set up logging
log = logging.getLogger("workbench")


def proximity_graph(prox: Proximity, n_neighbors: int = 5, min_edges: int = 2, min_weight: float = 0.8) -> nx.Graph:
    """Build a NetworkX graph from a Proximity instance.

    Nodes carry all columns of the proximity DataFrame; edges carry a `weight`
    (similarity, or distance normalized to [0, 1] for distance-based backends).

    Args:
        prox (Proximity): An instance of a Proximity class.
        n_neighbors (int): Number of neighbors to retrieve per node (default: 5).
        min_edges (int): Minimum edges per node (default: 2).
        min_weight (float): Weight threshold for additional edges beyond min_edges (default: 0.8).

    Returns:
        nx.Graph: The proximity graph.
    """
    node_df = prox.df
    id_column = prox.id_column

    # Get all neighbor pairs
    log.info("Retrieving all neighbors...")
    all_ids = node_df[id_column].tolist()
    neighbors_df = prox.neighbors(all_ids, n_neighbors=n_neighbors, include_self=False)

    # Handle duplicate IDs
    if not node_df[id_column].is_unique:
        log.warning(f"Column '{id_column}' contains duplicate values. Using first occurrence.")
        node_df = node_df.drop_duplicates(subset=[id_column], keep="first")

    log.info("Adding nodes to the proximity graph...")
    graph = nx.Graph()
    graph.add_nodes_from(node_df.set_index(id_column, drop=False).to_dict("index").items())

    # Compute edge weights (handle both distance-based and similarity-based proximity)
    if "similarity" in neighbors_df.columns:
        neighbors_df["weight"] = neighbors_df["similarity"]
    else:
        max_distance = neighbors_df["distance"].max()
        neighbors_df["weight"] = 1.0 - neighbors_df["distance"] / max_distance if max_distance > 0 else 1.0

    # Add edges: guarantee min_edges per node, plus any above min_weight
    log.info("Adding edges to the graph...")
    neighbors_df = neighbors_df.sort_values([id_column, "weight"], ascending=[True, False])
    rank = neighbors_df.groupby(id_column).cumcount()
    edges = neighbors_df[(rank < min_edges) | (neighbors_df["weight"] > min_weight)]
    graph.add_edges_from(zip(edges[id_column], edges["neighbor_id"], ({"weight": w} for w in edges["weight"])))

    return graph


if __name__ == "__main__":
    from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity
    from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

    # Quick test with feature data
    feature_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "Feature1": [0.1, 0.2, 0.3, 0.4],
            "Feature2": [0.5, 0.4, 0.3, 0.2],
            "target": [10, 20, 30, 40],
        }
    )

    print("--- FeatureSpaceProximity Graph ---")
    prox = FeatureSpaceProximity(feature_df, id_column="id", features=["Feature1", "Feature2"], target="target")
    graph = prox.graph(n_neighbors=3)
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    print("Edges:", graph.edges(data=True))

    # Quick test with fingerprint data
    fingerprint_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "fingerprint": ["101010", "111010", "101110", "011100"],
        }
    )

    print("\n--- FingerprintProximity Graph ---")
    prox = FingerprintProximity(fingerprint_df, fingerprint_column="fingerprint", id_column="id")
    graph = prox.graph(n_neighbors=3)
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    print("Edges:", graph.edges(data=True))

    # Neighborhood around a single node
    print("\n--- Neighborhood for Node 1 ---")
    neighborhood = nx.ego_graph(graph, 1, radius=1)
    print(f"Nodes: {list(neighborhood.nodes())}, Edges: {list(neighborhood.edges())}")

    # Real dataset with graph visualization
    from workbench.web_interface.components.plugins.graph_plot import GraphPlot
    from workbench.api import DFStore
    from workbench.utils.chem_utils.fingerprints import feature_fingerprints
    from workbench.utils.graph_utils import graph_layout

    print("\n--- Tox21 FingerprintProximity Graph ---")
    tox_df = DFStore().get("/datasets/chem_info/tox21")[:500]
    tox_df = feature_fingerprints(tox_df)
    prox = FingerprintProximity(tox_df, fingerprint_column="fingerprint", id_column="id")
    graph = prox.graph(n_neighbors=5)
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    # Largest connected component + layout
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph_layout(nx.subgraph(graph, largest_cc).copy())
    print(f"Largest CC - Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    # Show the graph
    graph_plot = GraphPlot()
    fig = graph_plot.update_properties(graph, label="id", hover_columns="all")[0]
    fig.update_layout(paper_bgcolor="rgb(30,30,30)", plot_bgcolor="rgb(30,30,30)")
    fig.show()
