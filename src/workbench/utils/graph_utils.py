"""Graph Utilities"""

import pandas as pd
import networkx as nx


def create_nxgraph_from_dfs(
    node_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_id_col: str = "node_id",
    edge_source_col: str = "source",
    edge_target_col: str = "target",
) -> nx.Graph:
    """
    Creates a NetworkX graph from node and edge DataFrames.

    Args:
        node_df (pd.DataFrame): DataFrame containing node attributes. Must include `node_id_col`.
        edges_df (pd.DataFrame): DataFrame containing edge information.
        node_id_col (str): Column name in `node_df` representing unique node IDs.
        edge_source_col (str): Column name in `edges_df` for edge source nodes.
        edge_target_col (str): Column name in `edges_df` for edge target nodes.

    Returns:
        nx.Graph: A NetworkX graph constructed from the input DataFrames.
    """
    # Create an empty graph
    G = nx.Graph()

    # Add nodes with attributes
    node_attributes = node_df.set_index(node_id_col).to_dict(orient="index")
    G.add_nodes_from([(node, attrs) for node, attrs in node_attributes.items()])

    # Add edges with attributes
    edge_attributes = edges_df.set_index([edge_source_col, edge_target_col]).to_dict(orient="index")
    G.add_edges_from([(src, tgt, attrs) for (src, tgt), attrs in edge_attributes.items()])

    return G


if __name__ == "__main__":
    # Test the graph utility functions

    # Example node DataFrame
    node_df = pd.DataFrame({"node_id": [1, 2, 3], "attribute": ["A", "B", "C"]})

    # Example edge DataFrame
    edges_df = pd.DataFrame({"source": [1, 2], "target": [2, 3], "weight": [0.5, 1.5]})

    # Create the networkx graph
    G = create_nxgraph_from_dfs(node_df, edges_df)

    # Check the graph
    print(G.nodes(data=True))
    print(G.edges(data=True))
