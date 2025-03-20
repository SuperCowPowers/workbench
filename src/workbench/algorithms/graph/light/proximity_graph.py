import networkx as nx
import pandas as pd
from typing import Union
import logging

# Workbench Imports
from workbench.algorithms.dataframe import Proximity, ProximityType
from workbench.api.graph_store import GraphStore

# Set up logging
log = logging.getLogger("workbench")


class ProximityGraph:
    """Build a NetworkX graph using the Proximity class."""

    def __init__(self):
        """Build a NetworkX graph using the Proximity class."""

        # The graph is stored as NetworkX graph
        self._nx_graph = None

        # GraphStore
        self.graph_store = GraphStore()

    def build_graph(self, proximity_instance: Proximity) -> None:
        """
        Build a NetworkX graph using a Proximity class.

        Args:
            proximity_instance (Proximity): An instance of a Proximity class to compute neighbors.
        """
        # Retrieve all neighbors and their distances
        prox = proximity_instance
        node_df = prox.df
        id_column = prox.id_column
        log.info("Retrieving all neighbors...")
        all_neighbors_df = prox.all_neighbors()

        # Add nodes with attributes (features)
        log.info("Adding nodes to the proximity graph...")
        self._nx_graph = nx.Graph()

        # Check for duplicate IDs in the node DataFrame
        if not node_df[id_column].is_unique:
            log.error(f"Column '{id_column}' contains duplicate values. Using first occurrence for each ID...")
            node_df = node_df.drop_duplicates(subset=[id_column], keep="first")

        # Set the id_column as index and add nodes
        self._nx_graph.add_nodes_from(node_df.set_index(id_column, drop=False).to_dict("index").items())

        # Determine edge weights based on proximity type
        if prox.proximity_type == ProximityType.SIMILARITY:
            all_neighbors_df["weight"] = all_neighbors_df["similarity"]
        elif prox.proximity_type == ProximityType.DISTANCE:
            # Normalize and invert distance
            max_distance = all_neighbors_df["distance"].max()
            all_neighbors_df["weight"] = 1.0 - all_neighbors_df["distance"] / max_distance

        # Add edges to the graph
        log.info("Adding edges to the graph...")
        min_edges = 2
        min_weight = 0.8

        # Group by source ID and process each group
        for source_id, group in all_neighbors_df.groupby(id_column):
            # Sort by weight (assuming higher is better)
            sorted_group = group.sort_values("weight", ascending=False)

            # Take all edges up to min_edges (or all if fewer)
            actual_min_edges = min(len(sorted_group), min_edges)
            top_edges = sorted_group.iloc[:actual_min_edges]

            # Also take any additional neighbors above min_weight (beyond the top edges)
            high_weight_edges = sorted_group.iloc[actual_min_edges:][
                sorted_group.iloc[actual_min_edges:]["weight"] > min_weight
            ]

            # Combine both sets
            edges_to_add = pd.concat([top_edges, high_weight_edges])

            # Add all edges at once
            self._nx_graph.add_edges_from(
                [(source_id, row["neighbor_id"], {"weight": row["weight"]}) for _, row in edges_to_add.iterrows()]
            )

    @property
    def nx_graph(self) -> nx.Graph:
        """
        Get the NetworkX graph object.

        Returns:
            nx.Graph: The NetworkX graph object.
        """
        return self._nx_graph

    def load_graph(self, graph_path: str):
        """
        Load a graph from the GraphStore.

        Args:
            graph_path (str): The path to the graph in GraphStore.
        """
        self._nx_graph = self.graph_store.get(graph_path)

    def store_graph(self, graph_path: str):
        """
        Store the graph in the GraphStore.

        Args:
            graph_path (str): The path to store the graph in GraphStore.
        """
        self.graph_store.upsert(graph_path, self._nx_graph)

    def get_neighborhood(self, node_id: Union[str, int], radius: int = 1) -> nx.Graph:
        """
        Get a subgraph containing nodes within a given number of hops around a specific node.

        Args:
            node_id: The ID of the node to center the neighborhood around.
            radius: The number of hops to consider around the node (default: 1).

        Returns:
            nx.Graph: A subgraph containing the specified neighborhood.
        """
        # Use NetworkX's ego_graph to extract the neighborhood within the given radius
        if node_id in self._nx_graph:
            return nx.ego_graph(self._nx_graph, node_id, radius=radius)
        else:
            raise ValueError(f"Node ID '{node_id}' not found in the graph.")


if __name__ == "__main__":
    from workbench.algorithms.dataframe.proximity import Proximity
    from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
    from workbench.web_interface.components.plugins.graph_plot import GraphPlot
    from workbench.api import DFStore
    from workbench.utils.chem_utils import compute_morgan_fingerprints, project_fingerprints
    from workbench.utils.graph_utils import connected_sample, graph_layout

    def show_graph(graph, id_column):
        """Display the graph using Plotly."""
        graph_plot = GraphPlot()
        properties = graph_plot.update_properties(graph, labels=id_column, hover_text="all")
        fig = properties[0]
        fig.update_layout(paper_bgcolor="rgb(30,30,30)", plot_bgcolor="rgb(30,30,30)")
        fig.show()

    # Example DataFrame for FeaturesProximity
    feature_data = {
        "id": [1, 2, 3, 4],
        "Feature1": [0.1, 0.2, 0.3, 0.4],
        "Feature2": [0.5, 0.4, 0.3, 0.2],
        "target": [10, 20, 30, 40],
    }
    feature_df = pd.DataFrame(feature_data)

    # Build a graph using the base Proximity class
    print("\n--- Proximity Class ---")
    prox = Proximity(feature_df, id_column="id", features=["Feature1", "Feature2"], target="target")
    feature_graph = ProximityGraph()
    feature_graph.build_graph(prox)
    nx_graph = feature_graph.nx_graph
    print("Edges:", nx_graph.edges(data=True))

    # Example DataFrame for FingerprintProximity
    fingerprint_data = {
        "id": [1, 2, 3, 4],
        "fingerprint": ["101010", "111010", "101110", "011100"],
    }
    fingerprint_df = pd.DataFrame(fingerprint_data)

    # Build a graph using FingerprintProximity
    print("\n--- FingerprintProximity Graph ---")
    prox = FingerprintProximity(fingerprint_df, fingerprint_column="fingerprint", id_column="id")
    fingerprint_graph = ProximityGraph()
    fingerprint_graph.build_graph(prox)
    nx_graph = fingerprint_graph.nx_graph
    print("Edges:", nx_graph.edges(data=True))

    # Neighborhood subgraph for FeaturesProximity
    print("\n--- Neighborhood Subgraph for Node 1 ---")
    neighborhood_subgraph = feature_graph.get_neighborhood(node_id=1, radius=1)
    print("Nodes:", neighborhood_subgraph.nodes())
    print("Edges:", neighborhood_subgraph.edges(data=True))

    # Plot the full graph
    show_graph(nx_graph, "id")

    # Get a neighborhood subgraph for a specific node
    neighborhood_subgraph = fingerprint_graph.get_neighborhood(node_id=fingerprint_df["id"].iloc[0], radius=2)

    # Plot the neighborhood subgraph
    show_graph(neighborhood_subgraph, "id")

    # Compute a shortest path subgraph using two random nodes
    source_node = fingerprint_df["id"].iloc[0]
    target_node = fingerprint_df["id"].iloc[-1]
    short_path = set(nx.shortest_path(nx_graph, source=source_node, target=target_node, weight="weight"))
    subgraph = nx_graph.subgraph(short_path)

    # Plot the subgraph
    show_graph(subgraph, "id")

    # Now a real dataset with fingerprints

    # Pull in the tox21 data
    tox_df = DFStore().get("/datasets/chem_info/tox21")[:1000]
    tox_df = compute_morgan_fingerprints(tox_df)
    id_column = "id"

    # Project the fingerprints to 2D space
    tox_df = project_fingerprints(tox_df, projection="UMAP")

    # Compute FingerprintProximity Graph
    print("\nComputing FingerprintProximity Graph for Tox21 Data...")
    prox = FingerprintProximity(tox_df, fingerprint_column="morgan_fingerprint", id_column=id_column, n_neighbors=5)
    fingerprint_graph = ProximityGraph()
    fingerprint_graph.build_graph(prox)
    nx_graph = fingerprint_graph.nx_graph
    print("\nTox21 Graph:")
    print("Nodes:", nx_graph.number_of_nodes())
    print("Edges:", nx_graph.number_of_edges())

    # Grab the biggest connected component
    largest_cc = max(nx.connected_components(nx_graph), key=len)
    nx_graph = nx.subgraph(nx_graph, largest_cc)
    print("Largest Connected Component:")
    print("Nodes:", nx_graph.number_of_nodes())
    print("Edges:", nx_graph.number_of_edges())

    # Now fine tune the projection with force-directed layout
    print("\nApplying force-directed layout...")
    nx_graph = graph_layout(nx_graph)

    # Store the graph in the GraphStore
    gstore = GraphStore()
    print("\nStoring the graph in GraphStore...")
    gstore.upsert("chem_info/tox21", nx_graph)

    # Compute a connected sample of the graph
    print("\nComputing a connected sample of the graph...")
    sample = connected_sample(nx_graph, n=100)
    sample = graph_layout(sample)

    # Store the graph in the GraphStore
    print("\nStoring the sample graph in GraphStore...")
    gstore.upsert("chem_info/tox21_100", sample)

    # Plot a sample of the graph
    print("\nShowing the connected sample graph...")
    show_graph(sample, id_column)
