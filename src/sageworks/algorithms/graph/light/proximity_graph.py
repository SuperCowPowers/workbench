import pandas as pd
import networkx as nx
from typing import Union

# SageWorks Imports
from sageworks.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity
from sageworks.utils.pandas_utils import drop_nans


class ProximityGraph:
    """
    Build a proximity graph of the nearest neighbors based on feature space.
    """

    def __init__(self, n_neighbors: int = 5):
        """Initialize the ProximityGraph with the specified parameters.

        Args:
            n_neighbors (int): Number of neighbors to consider (default: 5)
        """
        self.n_neighbors = n_neighbors
        self.nx_graph = nx.Graph()

    def build_graph(
        self,
        df: pd.DataFrame,
        features: list,
        id_column: str,
        target: str,
        store_features=True,
    ) -> nx.Graph:
        """
        Processes the input DataFrame and builds a proximity graph.

        Args:
            df (pd.DataFrame): The input DataFrame containing feature columns.
            features (list): List of feature column names to be used for building the proximity graph.
            id_column (str): Name of the ID column in the DataFrame.
            target (str): Name of the target column in the DataFrame.
            store_features (bool): Whether to store the features as node attributes (default: True).

        Returns:
            nx.Graph: The proximity graph as a NetworkX graph.
        """
        # Drop NaNs from the DataFrame using the provided utility
        df = drop_nans(df)

        # Initialize FeatureSpaceProximity with the input DataFrame and the specified features
        knn_spider = FeatureSpaceProximity(
            df,
            features=features,
            id_column=id_column,
            target=target,
            neighbors=self.n_neighbors,
        )

        # Use FeatureSpaceProximity to get all neighbor indices and distances
        indices, distances = knn_spider.get_neighbor_indices_and_distances()

        # Compute max distance for scaling (to [0, 1])
        max_distance = distances.max()

        # Initialize an empty graph
        self.nx_graph = nx.Graph()

        # Use the ID column for node IDs instead of relying on the DataFrame index
        node_ids = df[id_column].values

        # Add nodes with their features as attributes using the ID column
        for node_id in node_ids:
            if store_features:
                self.nx_graph.add_node(
                    node_id, **df[df[id_column] == node_id].iloc[0].to_dict()
                )  # Use .iloc[0] for correct node attributes
            else:
                self.nx_graph.add_node(node_id)

        # Add edges with weights based on inverse distance
        for i, neighbors in enumerate(indices):
            one_edge_added = False
            for j, neighbor_idx in enumerate(neighbors):
                if i != neighbor_idx:
                    # Compute the weight of the edge (inverse of distance)
                    weight = 1.0 - (distances[i][j] / max_distance)  # Scale to [0, 1]

                    # Map back to the ID column instead of the DataFrame index
                    src_node = node_ids[i]
                    dst_node = node_ids[neighbor_idx]

                    # Add the edge to the graph (if the weight is greater than 0.1)
                    if weight > 0.1 or not one_edge_added:
                        self.nx_graph.add_edge(src_node, dst_node, weight=weight)
                        one_edge_added = True

        # Return the NetworkX graph
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
    """Example usage of the ProximityGraph class"""
    from sageworks.api.feature_set import FeatureSet
    from sageworks.core.artifacts.graph_core import GraphCore
    from sageworks.web_components.plugins.graph_plot import GraphPlot

    # Load the Abalone FeatureSet
    fs = FeatureSet("abalone_features")
    df = fs.pull_dataframe()[:100]

    # Define the feature columns for the proximity graph
    feature_columns = [
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
    ]

    # Initialize the ProximityGraph with specified neighbors
    proximity_graph = ProximityGraph(n_neighbors=5)

    # Build the proximity graph using the specified features and ID column
    nx_graph = proximity_graph.build_graph(
        df, features=feature_columns, id_column=fs.id_column, target="class_number_of_rings"
    )

    # Create a SageWorks GraphCore object
    my_graph = GraphCore(nx_graph, "abalone_proximity_graph")
    print(my_graph.details())

    # Plot the full graph
    graph_plot = GraphPlot()
    properties = graph_plot.update_properties(my_graph, labels="id", hover_text="all")
    properties[0].show()

    # Get a neighborhood subgraph for a specific node
    neighborhood_subgraph = proximity_graph.get_neighborhood(node_id=df["id"].iloc[0], radius=2)

    # Plot the neighborhood subgraph
    properties = graph_plot.update_properties(neighborhood_subgraph, labels="id", hover_text="all")
    properties[0].show()

    # Compute a shortest path subgraph using two random nodes
    source_node = df["id"].iloc[0]
    target_node = df["id"].iloc[-1]
    nx_graph = my_graph.get_nx_graph()
    short_path = set(nx.shortest_path(nx_graph, source=source_node, target=target_node, weight="weight"))
    subgraph = nx_graph.subgraph(short_path)

    # Plot the subgraph
    graph_plot = GraphPlot()
    properties = graph_plot.update_properties(subgraph, labels="id", hover_text="all")
    properties[0].show()
