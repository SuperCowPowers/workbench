"""AWSGraphStore: Storage of NetworkX Graphs using AWS S3 and JSON serialization."""

import networkx as nx
import json
import logging
import re
from datetime import datetime
from typing import Union, Optional
from urllib.parse import urlparse
import pandas as pd

# Workbench Imports
from workbench.utils.config_manager import ConfigManager
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.json_utils import CustomEncoder, custom_decoder


class AWSGraphStore:
    """AWSGraphStore: Storage of NetworkX Graphs using AWS S3 and JSON serialization."""

    def __init__(self, path_prefix: Optional[str] = None):
        """AWSGraphStore Init Method

        Args:
            path_prefix (Optional[str]): Path prefix for storage locations (Defaults to None)
        """
        self.log = logging.getLogger("workbench")
        self._base_prefix = "graph_store/"
        self.path_prefix = self._normalize_path(self._base_prefix + (path_prefix or ""))

        # Retrieve bucket name and initialize S3 session
        config = ConfigManager()
        self.workbench_bucket = config.get_config("WORKBENCH_BUCKET")
        self.boto3_session = AWSAccountClamp().boto3_session
        self.s3_client = self.boto3_session.client("s3")

    def summary(self) -> pd.DataFrame:
        """Provide a summary of all graphs in the store.

        Returns:
            pd.DataFrame: Summary DataFrame with location, size, and modified date.
        """
        df = self.details()
        if df.empty:
            return pd.DataFrame(columns=["location", "size (MB)", "modified"])

        df["size (MB)"] = (df["size"] / (1024 * 1024)).round(2)
        df["modified"] = pd.to_datetime(df["modified"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return df[["location", "size (MB)", "modified"]]

    def details(self) -> pd.DataFrame:
        """Return detailed metadata for all stored graphs.

        Returns:
            pd.DataFrame: DataFrame with details like location, size, and last modified date.
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.workbench_bucket, Prefix=self.path_prefix)
            if "Contents" not in response:
                return pd.DataFrame(columns=["location", "s3_file", "size", "modified"])

            data = [
                {
                    "location": obj["Key"].replace(f"{self.path_prefix}", "").split(".json")[0],
                    "s3_file": f"s3://{self.workbench_bucket}/{obj['Key']}",
                    "size": obj["Size"],
                    "modified": obj["LastModified"],
                }
                for obj in response["Contents"]
            ]
            return pd.DataFrame(data)
        except Exception as e:
            self.log.error(f"Failed to get object details: {e}")
            return pd.DataFrame(columns=["location", "s3_file", "size", "modified"])

    def check(self, location: str) -> bool:
        """Check if a graph exists.

        Args:
            location (str): Logical location of the graph.

        Returns:
            bool: True if the graph exists, False otherwise.
        """
        s3_uri = self._generate_s3_uri(location)
        bucket, key = self._parse_s3_uri(s3_uri)

        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except self.s3_client.exceptions.ClientError:
            return False

    def get(self, location: str) -> Union[nx.Graph, None]:
        """Retrieve a NetworkX graph from AWS S3.

        Args:
            location (str): Logical location of the graph.

        Returns:
            Union[nx.Graph, None]: The retrieved graph or None if not found.
        """
        s3_uri = self._generate_s3_uri(location)
        bucket, key = self._parse_s3_uri(s3_uri)

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            graph_json = json.loads(response["Body"].read().decode("utf-8"), object_hook=custom_decoder)

            # Deserialize the graph
            graph = nx.readwrite.json_graph.node_link_graph(graph_json, edges="edges")

            # Replace "_node_id" back to "id" in the graph's node attributes
            for node in graph.nodes:
                if "_node_id" in graph.nodes[node]:
                    graph.nodes[node]["id"] = graph.nodes[node].pop("_node_id")
            return graph
        except json.JSONDecodeError as e:
            self.log.error(f"Failed to decode JSON for graph at '{s3_uri}': {e}")
            raise
        except Exception as e:
            self.log.error(f"Failed to retrieve graph from '{s3_uri}': {e}")
            return None

    def upsert(self, location: str, graph: nx.Graph):
        """Insert or update a NetworkX graph in AWS S3.

        Args:
            location (str): Logical location to store the graph.
            graph (nx.Graph): The NetworkX graph to store.
        """
        s3_uri = self._generate_s3_uri(location)
        bucket, key = self._parse_s3_uri(s3_uri)

        try:
            graph_json = nx.readwrite.json_graph.node_link_data(graph, edges="edges")

            # If we have an "id" field, replicate that into "_node_id" for serialization
            for node in graph_json["nodes"]:
                if "id" in node:
                    node["_node_id"] = node["id"]

            json_data = json.dumps(graph_json, cls=CustomEncoder)
            self.s3_client.put_object(Bucket=bucket, Key=key, Body=json_data)
            self.log.info(f"Graph stored at '{s3_uri}'")
        except Exception as e:
            self.log.error(f"Failed to store graph at '{s3_uri}': {e}")
            raise

    def list(self) -> list:
        """List all graphs in the store.

        Returns:
            list: A list of all graph locations in the store.
        """
        df = self.details()
        return df["location"].tolist()

    def last_modified(self, location: str) -> Union[datetime, None]:
        """Return the last modified date of a graph.

        Args:
            location (str): Logical location of the graph.

        Returns:
            Union[datetime, None]: Last modified datetime or None if not found.
        """
        s3_uri = self._generate_s3_uri(location)
        bucket, key = self._parse_s3_uri(s3_uri)

        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            return response["LastModified"]
        except self.s3_client.exceptions.ClientError:
            return None

    def delete(self, location: str):
        """Delete a NetworkX graph from AWS S3.

        Args:
            location (str): Logical location of the graph to delete.
        """
        s3_uri = self._generate_s3_uri(location)
        bucket, key = self._parse_s3_uri(s3_uri)

        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            self.log.info(f"Graph deleted from '{s3_uri}'")
        except Exception as e:
            self.log.error(f"Failed to delete graph from '{s3_uri}': {e}")

    def _generate_s3_uri(self, location: str) -> str:
        """Generate the S3 URI for the given location."""
        s3_path = f"{self.workbench_bucket}/{self.path_prefix}/{location}.json"
        return f"s3://{re.sub(r'/+', '/', s3_path)}"

    def _parse_s3_uri(self, s3_uri: str) -> tuple:
        """Parse an S3 URI into bucket and key."""
        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        return parsed.netloc, parsed.path.lstrip("/")

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize paths by collapsing slashes."""
        return re.sub(r"/+", "/", path)

    def __repr__(self):
        """Return a string representation of the AWSGraphStore object."""
        summary_df = self.summary()
        if summary_df.empty:
            return "AWSGraphStore: No graphs found in the store."
        return summary_df.to_string(index=False)


if __name__ == "__main__":
    """Exercise the AWSGraphStore Class"""
    # Create an AWSGraphStore instance
    graph_store = AWSGraphStore()

    # Create a test graph
    G = nx.erdos_renyi_graph(5, 0.5)
    for u, v, d in G.edges(data=True):
        d["weight"] = 1.0

    # Add node attributes, including an "id"
    for node in G.nodes:
        G.nodes[node]["id"] = node

    # Store the graph
    graph_store.upsert("test/test_graph", G)

    # Get the graph and print out attributes
    G = graph_store.get("test/test_graph")
    print("Graph Nodes:", G.nodes(data=True))

    # Get a summary of the graphs
    print("Graph Store Summary:")
    print(graph_store.summary())

    # Display detailed metadata
    print("Graph Store Details:")
    print(graph_store.details())

    # Last modified date
    print("Last modified:", graph_store.last_modified("test/test_graph"))

    # List all graphs
    print("List all graphs in the store:")
    print(graph_store.list())

    # Retrieve and display the graph
    retrieved_graph = graph_store.get("test/test_graph")
    print("Retrieved Graph Edges:", retrieved_graph.edges(data=True))

    # Check if the graph exists
    print("Graph exists:", graph_store.check("test/test_graph"))

    # Delete the graph
    graph_store.delete("test/test_graph")
    print("Graph exists after deletion:", graph_store.check("test/test_graph"))
