import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import networkx as nx

# SageWorks Imports
from sageworks.utils.pandas_utils import drop_nans


class ProximityGraph:
    """
    A custom transformer for building a proximity graph of the nearest neighbors based on feature space.

    Attributes:
        n_neighbors (int): Number of neighbors to consider.
    """

    def __init__(self, n_neighbors: int = 10):
        """Initialize the ProximityGraph with the specified parameters.

        Args:
            n_neighbors (int): Number of neighbors to consider (default: 10)
        """
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree", metric="euclidean")

    def build_graph(self, X: pd.DataFrame) -> nx.Graph:
        """
        Processes the input DataFrame and builds a proximity graph.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            nx.Graph: The proximity graph as a NetworkX graph.
        """
        X = drop_nans(X)

        # Standardize the features
        X_norm = self.scaler.fit_transform(X)

        # Fit the NearestNeighbors model
        self.nn_model.fit(X_norm)

        # Compute the nearest neighbors graph
        distances, indices = self.nn_model.kneighbors(X_norm)

        # Compute the max distances
        max_distance = distances.max()

        # Create the NetworkX graph
        G = nx.Graph()

        for i, neighbors in enumerate(indices):
            one_edge_added = False
            for j, neighbor in enumerate(neighbors):
                if i != neighbor:
                    # Compute the weight of the edge (inverse of the distance)
                    weight = 1.0 - (distances[i][j] / max_distance)

                    # Raising the weight to a power tends to emphasize the stronger connections
                    weight = weight ** 8

                    # Add the edge to the graph (if the weight is greater than 0.01)
                    if weight > 0.01 or not one_edge_added:
                        G.add_edge(i, neighbor, weight=weight)
                        one_edge_added = True

        return G


if __name__ == "__main__":
    """Example usage of the ProximityGraph class"""
    from sageworks.api.feature_set import FeatureSet
    from sageworks.api.model import Model

    # Load the Abalone FeatureSet
    fs = FeatureSet("abalone_features")
    df = fs.pull_dataframe()

    # Grab the feature columns from the model
    model = Model("abalone-regression")
    feature_columns = model.features()
    # feature_columns = ["length", "diameter"]

    # Initialize the ProximityGraph
    proximity_graph = ProximityGraph(n_neighbors=10)
    G = proximity_graph.build_graph(df[feature_columns])

    # Print the number of nodes and edges in the graph
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Compute the min and max edge weights
    edge_weights = [d["weight"] for u, v, d in G.edges(data=True)]
    print(f"Min Edge Weight: {min(edge_weights)}")
    print(f"Max Edge Weight: {max(edge_weights)}")

    # Pick a random node and print its features and its neighbors features
    node = 1
    print(f"Node {node} Features:")
    print(df[feature_columns].iloc[node])
    print(f"Number of Neighbors: {len(list(G.neighbors(node)))}")
    print(f"Node {node} Neighbors Features:")
    for neighbor in G.neighbors(node):
        # Edge Weights and Neighbor Features
        print(f"Neighbor: {neighbor} Edge Weight: {G[node][neighbor]['weight']}")
        print(df[feature_columns].iloc[neighbor])
