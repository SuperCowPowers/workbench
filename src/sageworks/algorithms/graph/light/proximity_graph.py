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
        X = drop_nans(X)

        # Standardize the features
        X_norm = self.scaler.fit_transform(X)

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
                graph.add_node(i, features=X.iloc[i].to_dict())
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
    from sageworks.api.model import Model
    import plotly.graph_objects as go

    # Load the Abalone FeatureSet
    fs = FeatureSet("abalone_features")
    df = fs.pull_dataframe()

    # Grab the feature columns from the model
    model = Model("abalone-regression")
    feature_columns = model.features()

    # Initialize the ProximityGraph
    proximity_graph = ProximityGraph(n_neighbors=5)
    my_graph = proximity_graph.build_graph(df[feature_columns])

    # Pick a specific node and print its features and its neighbors' features
    node = 1
    print(f"Node {node} Features:")
    print(my_graph.nodes[node]["features"])
    print(f"Number of Neighbors: {len(list(my_graph.neighbors(node)))}")
    print(f"Node {node} Neighbors Features:")

    # Get all neighbors and sort them by the edge weight
    neighbors_with_weights = [(neighbor, my_graph[node][neighbor]["weight"]) for neighbor in my_graph.neighbors(node)]
    sorted_neighbors = sorted(neighbors_with_weights, key=lambda x: x[1], reverse=True)

    for neighbor, weight in sorted_neighbors:
        # Edge Weights and Neighbor Features
        print(f"Neighbor: {neighbor} Edge Weight: {weight}")
        print(my_graph.nodes[neighbor]["features"])

    # Visualize a subgraph of the graph

    # Pick a specific node
    node_id = 1

    # Get nodes within 2 hops
    two_hop_neighbors = set(nx.single_source_shortest_path_length(my_graph, node_id, cutoff=2).keys())

    # Create a subgraph
    subgraph = my_graph.subgraph(two_hop_neighbors)

    # Get positions for the nodes
    pos = nx.spring_layout(subgraph)

    # Create edge traces
    edge_traces = []
    for edge in subgraph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]["weight"]
        width = weight * 4.9 + 0.1  # Scale to [0.1, 5]
        alpha = weight * 0.9 + 0.1  # Scale to [0.1, 1.0]

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=width, color=f"rgba(0, 0, 0, {alpha})"),
                hoverinfo="none",
                mode="lines",
            )
        )

    # Create node traces
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers+text",
        hoverinfo="text",
        marker=dict(showscale=False, color="skyblue", size=20, line_width=2),
    )

    for node in subgraph.nodes(data=True):
        x, y = pos[node[0]]
        node_trace["x"] += tuple([x])
        node_trace["y"] += tuple([y])
        node_trace["text"] += tuple([str(node[0])])

    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=f"Subgraph around node {node_id}",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(text="", showarrow=False, xref="paper", yref="paper")],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    fig.show()
