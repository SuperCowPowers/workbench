import networkx as nx
import pandas as pd
import logging

# Workbench Imports
from workbench.algorithms.dataframe.proximity import Proximity
from workbench.api.graph_store import GraphStore

# Set up logging
log = logging.getLogger("workbench")


class ProximityGraph:
    """Build a NetworkX graph using the Proximity class."""

    def __init__(self):
        """Build a NetworkX graph using the Proximity class."""
        self._nx_graph = None
        self.graph_store = GraphStore()

    def build_graph(self, prox: Proximity, n_neighbors: int = 5, min_edges: int = 2, min_weight: float = 0.8) -> None:
        """Build a NetworkX graph from a Proximity instance.

        Args:
            prox (Proximity): An instance of a Proximity class.
            n_neighbors (int): Number of neighbors to retrieve per node (default: 5).
            min_edges (int): Minimum edges per node (default: 2).
            min_weight (float): Weight threshold for additional edges beyond min_edges (default: 0.8).
        """
        node_df = prox.df
        id_column = prox.id_column

        # Get all neighbor pairs
        log.info("Retrieving all neighbors...")
        all_ids = node_df[id_column].tolist()
        neighbors_df = prox.neighbors(all_ids, n_neighbors=n_neighbors, include_self=False)

        # Build the graph and add nodes
        log.info("Adding nodes to the proximity graph...")
        self._nx_graph = nx.Graph()

        # Handle duplicate IDs
        if not node_df[id_column].is_unique:
            log.warning(f"Column '{id_column}' contains duplicate values. Using first occurrence.")
            node_df = node_df.drop_duplicates(subset=[id_column], keep="first")

        self._nx_graph.add_nodes_from(node_df.set_index(id_column, drop=False).to_dict("index").items())

        # Compute edge weights (handle both distance-based and similarity-based proximity)
        if "similarity" in neighbors_df.columns:
            neighbors_df["weight"] = neighbors_df["similarity"]
        else:
            max_distance = neighbors_df["distance"].max()
            neighbors_df["weight"] = 1.0 - neighbors_df["distance"] / max_distance if max_distance > 0 else 1.0

        # Add edges: guarantee min_edges per node, plus any above min_weight
        log.info("Adding edges to the graph...")
        for source_id, group in neighbors_df.groupby(id_column):
            sorted_group = group.sort_values("weight", ascending=False)
            n_top = min(len(sorted_group), min_edges)
            top_edges = sorted_group.iloc[:n_top]
            extra_edges = sorted_group.iloc[n_top:]
            extra_edges = extra_edges[extra_edges["weight"] > min_weight]
            edges = pd.concat([top_edges, extra_edges])
            self._nx_graph.add_edges_from(
                [(source_id, row["neighbor_id"], {"weight": row["weight"]}) for _, row in edges.iterrows()]
            )

    @property
    def nx_graph(self) -> nx.Graph:
        """Get the NetworkX graph object.

        Returns:
            nx.Graph: The NetworkX graph object.
        """
        return self._nx_graph

    def load_graph(self, graph_path: str):
        """Load a graph from the GraphStore.

        Args:
            graph_path (str): The path to the graph in GraphStore.
        """
        self._nx_graph = self.graph_store.get(graph_path)

    def store_graph(self, graph_path: str):
        """Store the graph in the GraphStore.

        Args:
            graph_path (str): The path to store the graph in GraphStore.
        """
        self.graph_store.upsert(graph_path, self._nx_graph)

    def get_neighborhood(self, node_id, radius: int = 1) -> nx.Graph:
        """Get a subgraph containing nodes within a given number of hops around a node.

        Args:
            node_id: The ID of the center node.
            radius (int): Number of hops (default: 1).

        Returns:
            nx.Graph: A subgraph containing the neighborhood.
        """
        if node_id not in self._nx_graph:
            raise ValueError(f"Node ID '{node_id}' not found in the graph.")
        return nx.ego_graph(self._nx_graph, node_id, radius=radius)


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
    graph = ProximityGraph()
    graph.build_graph(prox, n_neighbors=3)
    print(f"Nodes: {graph.nx_graph.number_of_nodes()}, Edges: {graph.nx_graph.number_of_edges()}")
    print("Edges:", graph.nx_graph.edges(data=True))

    # Quick test with fingerprint data
    fingerprint_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "fingerprint": ["101010", "111010", "101110", "011100"],
        }
    )

    print("\n--- FingerprintProximity Graph ---")
    prox = FingerprintProximity(fingerprint_df, fingerprint_column="fingerprint", id_column="id")
    graph = ProximityGraph()
    graph.build_graph(prox, n_neighbors=3)
    print(f"Nodes: {graph.nx_graph.number_of_nodes()}, Edges: {graph.nx_graph.number_of_edges()}")
    print("Edges:", graph.nx_graph.edges(data=True))

    # Neighborhood test
    print("\n--- Neighborhood for Node 1 ---")
    neighborhood = graph.get_neighborhood(node_id=1, radius=1)
    print(f"Nodes: {list(neighborhood.nodes())}, Edges: {list(neighborhood.edges())}")

    # Real dataset with graph visualization
    from workbench.web_interface.components.plugins.graph_plot import GraphPlot
    from workbench.api import DFStore
    from workbench.utils.chem_utils.fingerprints import compute_morgan_fingerprints
    from workbench.utils.graph_utils import graph_layout

    print("\n--- Tox21 FingerprintProximity Graph ---")
    tox_df = DFStore().get("/datasets/chem_info/tox21")[:500]
    tox_df = compute_morgan_fingerprints(tox_df)
    prox = FingerprintProximity(tox_df, fingerprint_column="fingerprint", id_column="id")
    graph = ProximityGraph()
    graph.build_graph(prox, n_neighbors=5)
    nx_graph = graph.nx_graph
    print(f"Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}")

    # Largest connected component + layout
    largest_cc = max(nx.connected_components(nx_graph), key=len)
    nx_graph = nx.subgraph(nx_graph, largest_cc).copy()
    nx_graph = graph_layout(nx_graph)
    print(f"Largest CC - Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}")

    # Show the graph
    graph_plot = GraphPlot()
    props = graph_plot.update_properties(nx_graph, labels="id", hover_text="all")
    fig = props[0]
    fig.update_layout(paper_bgcolor="rgb(30,30,30)", plot_bgcolor="rgb(30,30,30)")
    fig.show()
