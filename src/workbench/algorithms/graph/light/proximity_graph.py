import networkx as nx
import pandas as pd
from typing import Union
from workbench.algorithms.dataframe import Proximity, FeaturesProximity
import logging

# Set up logging
log = logging.getLogger("workbench")


class ProximityGraph:
    """
    Build a NetworkX graph using the Proximity class.
    """

    def __init__(
        self,
        proximity_instance: Proximity,
        store_features: bool = True,
    ):
        """
        Build a NetworkX graph using the Proximity class.

        Args:
            proximity_instance (Proximity): An instance of a Proximity class to compute neighbors.
            store_features (bool): Whether to store the features as node attributes (default: True).
        """

        # Initialize the NetworkX graph
        self.nx_graph = nx.Graph()

        # Handle to our Proximity class instance
        self.proximity = proximity_instance

        # Build the graph using the Proximity class
        self._build_graph(store_features)

    def _build_graph(self, store_features: bool) -> None:
        """
        Build a NetworkX graph using the Proximity class.

        Args:
            store_features (bool): Whether to store features as node attributes.
        """
        # Retrieve all neighbors and their distances
        id_column = self.proximity.id_column
        log.info("Retrieving all neighbors...")
        all_neighbors_df = self.proximity.all_neighbors()

        # Add nodes with attributes (features)
        log.info("Adding nodes to the graph...")
        for _, row in all_neighbors_df.iterrows():
            node_id = row[id_column]
            if store_features:
                self.nx_graph.add_node(node_id, **row.to_dict())
            else:
                self.nx_graph.add_node(node_id)

        # Add edges with weights based on proximity
        min_edges = 2
        min_wieght = 0.8
        current_id = None
        log.info("Adding edges to the graph...")
        for _, row in all_neighbors_df.iterrows():
            source_id = row[id_column]
            if source_id != current_id:
                num_edges = 0
                current_id = source_id
            weight = self.proximity.get_edge_weight(row)
            if num_edges <= min_edges or weight > min_wieght:
                weight = 0.1 if weight < 0.1 else weight
                self.nx_graph.add_edge(row[id_column], row["neighbor_id"], weight=weight)
                num_edges += 1

        # Print the number of nodes and edges
        nodes = self.nx_graph.number_of_nodes()
        edges = self.nx_graph.number_of_edges()
        log.info(f"Graph built with {nodes} nodes and {edges} edges.")

    def get_graph(self) -> nx.Graph:
        """
        Get the NetworkX graph object.

        Returns:
            nx.Graph: The NetworkX graph object.
        """
        return self.nx_graph

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
        if node_id in self.nx_graph:
            return nx.ego_graph(self.nx_graph, node_id, radius=radius)
        else:
            raise ValueError(f"Node ID '{node_id}' not found in the graph.")


if __name__ == "__main__":
    from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
    from workbench.web_interface.components.plugins.graph_plot import GraphPlot

    # Example DataFrame for FeaturesProximity
    feature_data = {
        "id": [1, 2, 3, 4],
        "Feature1": [0.1, 0.2, 0.3, 0.4],
        "Feature2": [0.5, 0.4, 0.3, 0.2],
        "target": [10, 20, 30, 40],
    }
    feature_df = pd.DataFrame(feature_data)

    # Build a graph using FeaturesProximity
    print("\n--- FeaturesProximity Graph ---")
    prox = FeaturesProximity(
        feature_df, id_column="id", features=["Feature1", "Feature2"], n_neighbors=2, target="target"
    )
    feature_graph = ProximityGraph(prox)
    nx_graph = feature_graph.get_graph()
    print("Edges:", nx_graph.edges(data=True))

    # Example DataFrame for FingerprintProximity
    fingerprint_data = {
        "id": [1, 2, 3, 4],
        "fingerprint": ["101010", "111010", "101110", "011100"],
    }
    fingerprint_df = pd.DataFrame(fingerprint_data)

    # Build a graph using FingerprintProximity
    print("\n--- FingerprintProximity Graph ---")
    prox = FingerprintProximity(fingerprint_df, fingerprint_column="fingerprint", id_column="id", n_neighbors=2)
    fingerprint_graph = ProximityGraph(prox)
    nx_graph = fingerprint_graph.get_graph()
    print("Edges:", nx_graph.edges(data=True))

    # Neighborhood subgraph for FeaturesProximity
    print("\n--- Neighborhood Subgraph for Node 1 ---")
    neighborhood_subgraph = feature_graph.get_neighborhood(node_id=1, radius=1)
    print("Nodes:", neighborhood_subgraph.nodes())
    print("Edges:", neighborhood_subgraph.edges(data=True))

    # Plot the full graph
    id_column = "id"
    graph_plot = GraphPlot()
    properties = graph_plot.update_properties(nx_graph, labels=id_column, hover_text="all")
    properties[0].show()

    # Get a neighborhood subgraph for a specific node
    neighborhood_subgraph = fingerprint_graph.get_neighborhood(node_id=fingerprint_df[id_column].iloc[0], radius=2)

    # Plot the neighborhood subgraph
    properties = graph_plot.update_properties(neighborhood_subgraph, labels=id_column, hover_text="all")
    properties[0].show()

    # Compute a shortest path subgraph using two random nodes
    source_node = fingerprint_df[id_column].iloc[0]
    target_node = fingerprint_df[id_column].iloc[-1]
    short_path = set(nx.shortest_path(nx_graph, source=source_node, target=target_node, weight="weight"))
    subgraph = nx_graph.subgraph(short_path)

    # Plot the subgraph
    properties = graph_plot.update_properties(subgraph, labels=id_column, hover_text="all")
    properties[0].show()

    # Now a real dataset with fingerprints
    from workbench.api import FeatureSet
    from workbench.utils.chem_utils import compute_morgan_fingerprints

    fs = FeatureSet("aqsol_mol_descriptors")
    df = fs.pull_dataframe()
    df = df.sample(1000)
    df = compute_morgan_fingerprints(df)

    # Build a graph using FingerprintProximity
    print("\n--- FingerprintProximity Graph ---")
    prox = FingerprintProximity(df, fingerprint_column="morgan_fingerprint", id_column=fs.id_column, n_neighbors=5)
    fingerprint_graph = ProximityGraph(prox)
    nx_graph = fingerprint_graph.get_graph()

    # Plot the full graph
    id_column = fs.id_column
    graph_plot = GraphPlot()
    properties = graph_plot.update_properties(nx_graph, labels=id_column, hover_text="all")
    properties[0].show()

    # Save the graph to the GraphStore
    from workbench.api import GraphStore

    g_store = GraphStore()
    g_store.upsert("test/fingerprint_graph", nx_graph)
