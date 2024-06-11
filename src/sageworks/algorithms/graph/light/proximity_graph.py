import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import networkx as nx

# SageWorks Imports
from sageworks.utils.pandas_utils import drop_nans


class ProximityGraph:
    """
    Build a proximity graph of the nearest neighbors based on feature space.

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

    def build_graph(self, X: pd.DataFrame, store_features=True) -> nx.Graph:
        """
        Processes the input DataFrame and builds a proximity graph.

        Args:
            X (pd.DataFrame): The input features.
            store_features (bool): Whether to store the features as node attributes (default: True).

        Returns:
            nx.Graph: The proximity graph as a NetworkX graph.
        """
        # Drop any NaN values
        X = drop_nans(X)

        # Select only numeric columns
        numeric_cols = X.select_dtypes(include=["number"]).columns
        X_numeric = X[numeric_cols]

        # Standardize the features
        X_norm = self.scaler.fit_transform(X_numeric)

        # Fit the NearestNeighbors model
        self.nn_model.fit(X_norm)

        # Compute the nearest neighbors graph
        distances, indices = self.nn_model.kneighbors(X_norm)

        # Compute max distance for scaling
        max_distance = distances.max()

        # Create the NetworkX graph
        graph = nx.Graph()

        # Add nodes with their features as attributes
        if store_features:
            for i in range(X.shape[0]):
                graph.add_node(i, **X.iloc[i].to_dict())
        else:
            for i in range(X.shape[0]):
                graph.add_node(i)

        # Add edges with weights based on inverse distance
        for i, neighbors in enumerate(indices):
            one_edge_added = False
            for j, neighbor in enumerate(neighbors):
                if i != neighbor:
                    # Compute the weight of the edge (inverse of distance)
                    weight = 1.0 - (distances[i][j] / max_distance)  # Scale to [0, 1]

                    # Raising the weight to a power tends give better proximity weights
                    weight = weight**10

                    # Add the edge to the graph (if the weight is greater than 0.01)
                    if weight > 0.1 or not one_edge_added:
                        graph.add_edge(i, neighbor, weight=weight)
                        one_edge_added = True

        # Return the graph
        return graph


if __name__ == "__main__":
    """Example usage of the ProximityGraph class"""
    from sageworks.api.feature_set import FeatureSet
    from sageworks.core.artifacts.graph_core import GraphCore
    from sageworks.web_components.plugins.graph_plot import GraphPlot

    # Load the Abalone FeatureSet
    fs = FeatureSet("abalone_features")
    df = fs.pull_dataframe()

    # Drop any columns generated from AWS
    aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time", "training"]
    df = df.drop(columns=aws_cols, errors="ignore")

    # Initialize the ProximityGraph
    proximity_graph = ProximityGraph(n_neighbors=5)
    nx_graph = proximity_graph.build_graph(df)

    # Create a SageWorks GraphCore object
    my_graph = GraphCore(nx_graph, "abalone_proximity_graph")
    print(my_graph.details())

    # Grab a subgraph of the graph
    nx_graph = my_graph.get_nx_graph()
    two_hop_neighbors = set(nx.single_source_shortest_path_length(nx_graph, 0, cutoff=2).keys())
    subgraph = nx_graph.subgraph(two_hop_neighbors)

    # Plot the subgraph
    graph_plot = GraphPlot()
    [fig] = graph_plot.update_properties(subgraph, labels="id", hover_text="all")
    fig.show()
