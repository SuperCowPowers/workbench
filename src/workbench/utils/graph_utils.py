"""Graph/NetworkX Utilities"""

import pandas as pd
import json
from typing import Optional
from datetime import datetime
import logging

# NetworkX import
try:
    import networkx as nx
except ImportError:
    print("NetworkX Python module not found! pip install networkx")
    raise ImportError("NetworkX Python module not found! pip install networkx")

# Workbench imports
from workbench.utils.json_utils import CustomEncoder

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


def exists(graph_name: str) -> bool:
    """Check if the graph exists in S3
    
    Args:
        graph_name (str): The name of the graph artifact
    Returns:
        bool: True if the graph exists, False otherwise
    """
    s3 = s3_client()
    try:
        s3.head_object(Bucket=graph_bucket(), Key=f"graphs/{graph_name}.json")
        return True
    except s3.exceptions.ClientError:
        return False


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
    graph.graph['tags'] = tags


def get_tags(graph) -> list:
    """Get the tags for this graph artifact

    Args:
        graph (nx.Graph): The NetworkX graph artifact

    Returns:
        list: A list of tags associated with the graph artifact
    """
    return graph.graph.get('tags', [])


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


def delete(graph):
    """Delete this graph artifact including all related AWS objects"""
    s3_client().delete_object(Bucket=graph_bucket(), Key=f"graphs/{graph.name}.json")
    log.info(f"Graph {graph.name} deleted from S3")


def save(graph) -> None:
    """Save the internal NetworkX graph to S3"""
    graph_json = nx.readwrite.json_graph.node_link_data(graph)
    graph_str = json.dumps(graph_json, cls=CustomEncoder)
    s3_client().put_object(Bucket=graph_bucket(), Key=f"graphs/{graph.name}.json", Body=graph_str)
    log.info(f"Graph {graph.name} saved to S3")


def load_graph(graph_name: str) -> nx.Graph:
    """Load a NetworkX graph from S3"""

    bucket = graph_bucket()
    key = f"graphs/{graph_name}.json"

    # Load the graph from S3
    response = s3_client().get_object(Bucket=bucket, Key=key)
    graph_str = response["Body"].read().decode("utf-8")
    graph_json = json.loads(graph_str)
    return nx.readwrite.json_graph.node_link_graph(graph_json)


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

    # Load a graph from S3
    if exists("karate_club"):
        G = load_graph("karate_club")
    else:
        print("Graph 'karate_club' not found in S3")
        sys.exit(1)

    # Set description and tags for the graph
    G.graph["description"] = "Zachary's Karate Club"
    set_tags(G, ["social", "karate"])
    print(f"Tags: {get_tags(G)}")
    save(G)

    # Load the graph
    G = load_graph("karate_club")
    print(details(G))

    # View the graph
    graph_plot = GraphPlot()
    [figure, *_] = graph_plot.update_properties(G)
    figure.show()
