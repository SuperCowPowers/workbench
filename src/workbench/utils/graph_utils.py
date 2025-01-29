"""Graph/NetworkX Utilities

Note: For most end-users the GraphStore() class is the API to use
"""

import pandas as pd
import json
from typing import Optional
from datetime import datetime
import logging
import random

# NetworkX import
try:
    import networkx as nx
except ImportError:
    print("NetworkX Python module not found! pip install networkx")
    raise ImportError("NetworkX Python module not found! pip install networkx")

# Set up logging
log = logging.getLogger("workbench")


def graph_bucket() -> Optional[str]:
    """Return the S3 bucket for storing graph artifacts

    Returns:
        Optional[str]: The S3 bucket name for storing graph
    """
    from workbench.utils.config_manager import ConfigManager

    cm = ConfigManager()
    if not cm.config_okay():
        log.error("Workbench ConfigManager not initialized")
        return None
    return cm.get_config("WORKBENCH_BUCKET")


def s3_client():
    """Return the S3 client for storing/querying graph artifacts"""
    from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

    aws_account_clamp = AWSAccountClamp()
    boto3_session = aws_account_clamp.boto3_session
    return boto3_session.client("s3")


def details(graph) -> dict:
    """Additional details about this graph

    Returns:
        dict: A dictionary of details about the graph
    """
    return {
        "description": graph.graph.get("description", ""),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "size": size(graph),
        "tags": get_tags(graph),
    }


def size(graph) -> float:
    """Return the size of this graph artifact in MegaBytes"""
    graph_str = json.dumps(nx.readwrite.json_graph.node_link_data(graph))
    return len(graph_str.encode("utf-8")) / (1024 * 1024)


def modified(graph) -> Optional[datetime]:
    """Get the last modified date for this graph artifact

    Returns:
        Optional[datetime]: The last modified date of the graph artifact
    """
    bucket = graph_bucket()
    if not bucket:
        log.error("No S3 bucket configured for graph storage")
        return None
    s3 = s3_client()
    response = s3.head_object(Bucket=bucket, Key=f"graphs/{graph.name}.json")
    return response["LastModified"]


def set_tags(graph, tags: list) -> None:
    """Set the tags for this graph artifact

    Args:
        graph (nx.Graph): The NetworkX graph artifact
        tags (list): A list of tags to associate with the graph artifact
    """
    # Store tags as a graph attribute
    graph.graph["tags"] = tags


def get_tags(graph) -> list:
    """Get the tags for this graph artifact

    Args:
        graph (nx.Graph): The NetworkX graph artifact

    Returns:
        list: A list of tags associated with the graph artifact
    """
    return graph.graph.get("tags", [])


def arn(graph) -> Optional[str]:
    """AWS ARN (Amazon Resource Name) for this graph artifact

    Returns:
        Optional[str]: The AWS ARN for the graph artifact
    """
    return f"arn:aws:s3:::{graph_bucket()}/graphs/{graph.name}.json"


def aws_url(graph) -> Optional[str]:
    """AWS console/web interface for this graph artifact

    Returns:
        Optional[str]: The AWS URL for the graph artifact
    """
    return f"https://s3.console.aws.amazon.com/s3/object/{graph_bucket()}?prefix=graphs/{graph.name}.json"


def load_graph_from_file(file_path: str) -> Optional[nx.Graph]:
    """Load a graph from a file path

    Args:
        file_path (str): The path to the file containing the graph

    Returns:
        Optional[nx.Graph]: The NetworkX graph loaded from the file
    """
    try:
        with open(file_path, "r") as file:
            graph_json = json.load(file)
        return nx.readwrite.json_graph.node_link_graph(graph_json)
    except FileNotFoundError:
        log.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        log.error(f"Error decoding JSON from file: {file_path}")
        return None
    except Exception as e:
        log.error(f"An error occurred while loading the graph from file: {file_path}. Error: {e}")
        return None


def connected_sample(G: nx.Graph, n: int, remove_pos: bool = True) -> nx.Graph:
    """Sample a connected subgraph of a given size from a NetworkX graph.

    Args:
        G (nx.Graph): The input NetworkX graph.
        n (int): The number of nodes to sample.
        remove_pos (bool): Remove node positions if they exist (default: True).
    """
    # Get the largest connected component
    largest_component = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_component)

    # Ensure n doesn't exceed the number of nodes in the component
    n = min(n, len(subgraph.nodes))

    # Start from a random node
    start_node = random.choice(list(subgraph.nodes))
    sampled_nodes = {start_node}

    # Perform a breadth-first sampling until we have n nodes
    queue = [start_node]
    while queue and len(sampled_nodes) < n:
        current_node = queue.pop(0)
        neighbors = list(subgraph.neighbors(current_node))
        random.shuffle(neighbors)  # Randomize neighbor order
        for neighbor in neighbors:
            if neighbor not in sampled_nodes and len(sampled_nodes) < n:
                sampled_nodes.add(neighbor)
                queue.append(neighbor)

    # Induce a subgraph on the sampled nodes
    sampled_subgraph = subgraph.subgraph(sampled_nodes)

    # Remove node positions if they exist
    if remove_pos:
        for node in sampled_subgraph.nodes:
            if "pos" in sampled_subgraph.nodes[node]:
                del sampled_subgraph.nodes[node]["pos"]
    return sampled_subgraph


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
    import sys
    from workbench.api import GraphStore
    from workbench.web_interface.components.plugins.graph_plot import GraphPlot

    # Example node DataFrame
    node_df = pd.DataFrame({"node_id": [1, 2, 3], "attribute": ["A", "B", "C"]})

    # Example edge DataFrame
    edges_df = pd.DataFrame({"source": [1, 2], "target": [2, 3], "weight": [0.5, 1.5]})

    # Create the networkx graph
    G = create_nxgraph_from_dfs(node_df, edges_df)

    # Check the graph
    print(G.nodes(data=True))
    print(G.edges(data=True))

    # Load a graph from the GraphStore
    graph_store = GraphStore()
    if graph_store.check("test/karate_club"):
        G = graph_store.get("test/karate_club")
    else:
        print("Graph 'karate_club' not found in the GraphStore")
        sys.exit(1)

    # Set description and tags for the graph
    G.graph["description"] = "Zachary's Karate Club"
    set_tags(G, ["social", "karate"])
    print(f"Tags: {get_tags(G)}")

    # Show details
    print(details(G))

    # View the graph
    graph_plot = GraphPlot()
    [figure, *_] = graph_plot.update_properties(G)
    figure.show()

    # Sample the graph
    s_graph = connected_sample(G, 10)
    print(f"Sampled Graph: {s_graph.nodes()}")
    [figure, *_] = graph_plot.update_properties(s_graph)
    figure.show()
