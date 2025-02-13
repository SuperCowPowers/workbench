import networkx as nx
import pandas as pd
from typing import Union
import logging

# Workbench Imports
from workbench.algorithms.dataframe import Proximity, FeaturesProximity  # noqa: F401
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

    def build_graph(self, proximity_instance: Proximity, node_attributes_df: pd.DataFrame) -> None:
        """
        Build a NetworkX graph using a Proximity class.

        Args:
            proximity_instance (Proximity): An instance of a Proximity class to compute neighbors.
            node_attributes_df (pd.DataFrame): DataFrame containing node attributes
        """
        # Retrieve all neighbors and their distances
        prox = proximity_instance
        id_column = prox.id_column
        log.info("Retrieving all neighbors...")
        all_neighbors_df = prox.all_neighbors()

        # Add nodes with attributes (features)
        log.info("Building proximity graph...")
        self._nx_graph = nx.Graph()
        for _, row in all_neighbors_df.iterrows():
            node_id = row[id_column]

            # Get all the node attributes
            if node_id in node_attributes_df[id_column].values:
                node_attributes = node_attributes_df[node_attributes_df[id_column] == node_id].iloc[0].to_dict()
                self._nx_graph.add_node(node_id, **node_attributes)
            else:
                log.error(f"Node ID '{node_id}' not found in the node attributes DataFrame. Terminating graph build.")
                return

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
            weight = prox.get_edge_weight(row)
            if num_edges <= min_edges or weight > min_wieght:
                weight = 0.1 if weight < 0.1 else weight
                self._nx_graph.add_edge(row[id_column], row["neighbor_id"], weight=weight)
                num_edges += 1

        # Print the number of nodes and edges
        nodes = self._nx_graph.number_of_nodes()
        edges = self._nx_graph.number_of_edges()
        log.info(f"Graph built with {nodes} nodes and {edges} edges.")

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
    from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
    from workbench.web_interface.components.plugins.graph_plot import GraphPlot
    from workbench.api import DFStore
    from workbench.utils.chem_utils import compute_morgan_fingerprints
    from workbench.utils.graph_utils import connected_sample

    """
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
    prox = FingerprintProximity(fingerprint_df, fingerprint_column="fingerprint", id_column="id", n_neighbors=2)
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
    """

    # Now a real dataset with fingerprints

    # Pull in the tox21 data
    tox_df = DFStore().get("/datasets/chem_info/tox21")[:1000]
    tox_df = compute_morgan_fingerprints(tox_df)
    id_column = "id"

    # Compute FingerprintProximity Graph
    prox = FingerprintProximity(tox_df, fingerprint_column="morgan_fingerprint", id_column=id_column, n_neighbors=5)
    fingerprint_graph = ProximityGraph()
    fingerprint_graph.build_graph(prox, tox_df)
    nx_graph = fingerprint_graph.nx_graph

    # Store the graph in the GraphStore
    fingerprint_graph.store_graph("chem_info/tox21")

    # Plot a sample of the graph
    sample = connected_sample(nx_graph, n=100)
    graph_plot = GraphPlot()
    properties = graph_plot.update_properties(sample, labels=id_column, hover_text="all")
    properties[0].show()

    # Store the graph and load it back
    graph_store = GraphStore()
    graph_store.upsert("chem_info/tox21_100", sample)
    load_sample = graph_store.get("chem_info/tox21_100")

    # Plot to compare
    properties = graph_plot.update_properties(load_sample, labels=id_column, hover_text="all")
    properties[0].show()
