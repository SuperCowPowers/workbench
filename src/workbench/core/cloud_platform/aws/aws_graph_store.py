"""AWSGraphStore: Storage of NetworkX Graphs using AWS S3 and JSON serialization."""

import networkx as nx
import json
import logging
import re
from datetime import datetime, timezone
from typing import Union
import pandas as pd
import boto3

# Workbench Imports
from workbench.utils.config_manager import ConfigManager
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.aws_utils import not_found_returns_none


class AWSGraphStore:
    """AWSGraphStore: Storage of NetworkX Graphs using AWS S3 and JSON serialization."""

    def __init__(self, path_prefix: Union[str, None] = None):
        """AWSGraphStore Init Method

        Args:
            path_prefix (Union[str, None], optional): Path prefix for storage locations (Defaults to None)
        """
        self.log = logging.getLogger("workbench")
        self._base_prefix = "graph_store/"
        self.path_prefix = self._base_prefix + path_prefix if path_prefix else self._base_prefix
        self.path_prefix = re.sub(r"/+", "/", self.path_prefix)  # Collapse slashes

        # Retrieve bucket name and initialize S3 session
        config = ConfigManager()
        self.workbench_bucket = config.get_config("WORKBENCH_BUCKET")
        self.boto3_session = AWSAccountClamp().boto3_session
        self.s3_client = self.boto3_session.client("s3")

    def list(self) -> list:
        """List all graphs in the store."""
        df = self.details()
        return df["location"].tolist()

    def last_modified(self, location: str) -> Union[datetime, None]:
        """Return the last modified date of a graph."""
        df = self.details()
        mask = df["location"] == location

        if mask.any():
            time_str = df.loc[mask, "modified"].values[0]
            time_obj = pd.to_datetime(time_str)
            return time_obj.to_pydatetime().replace(tzinfo=timezone.utc)
        return None

    def summary(self) -> pd.DataFrame:
        """Provide a summary of all graphs in the store."""
        df = self.details()
        if df.empty:
            return pd.DataFrame(columns=["location", "size (MB)", "modified"])

        df["size (MB)"] = (df["size"] / (1024 * 1024)).round(2)
        df["modified"] = pd.to_datetime(df["modified"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return df[["location", "size (MB)", "modified"]]

    def details(self) -> pd.DataFrame:
        """Return detailed metadata for all stored graphs."""
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
        """Check if a graph exists."""
        s3_uri = self._generate_s3_uri(location)
        try:
            self.s3_client.head_object(Bucket=self.workbench_bucket, Key=self._parse_s3_uri(s3_uri)[1])
            return True
        except self.s3_client.exceptions.ClientError:
            return False

    @not_found_returns_none
    def get(self, location: str) -> Union[nx.Graph, None]:
        """Retrieve a NetworkX graph from AWS S3."""
        s3_uri = self._generate_s3_uri(location)
        bucket, key = self._parse_s3_uri(s3_uri)

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            json_data = response["Body"].read().decode("utf-8")
            graph_json = json.loads(json_data)
            return nx.readwrite.json_graph.node_link_graph(graph_json)
        except Exception as e:
            self.log.error(f"Failed to retrieve graph from '{s3_uri}': {e}")
            return None

    def upsert(self, location: str, graph: nx.Graph):
        """Insert or update a NetworkX graph in AWS S3."""
        s3_uri = self._generate_s3_uri(location)
        bucket, key = self._parse_s3_uri(s3_uri)

        try:
            graph_json = nx.readwrite.json_graph.node_link_data(graph)
            json_data = json.dumps(graph_json, indent=2)
            self.s3_client.put_object(Bucket=bucket, Key=key, Body=json_data)
            self.log.info(f"Graph stored at '{s3_uri}'")
        except Exception as e:
            self.log.error(f"Failed to store graph at '{s3_uri}': {e}")
            raise

    def delete(self, location: str):
        """Delete a NetworkX graph from AWS S3."""
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
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        _, _, bucket_and_key = s3_uri.partition("s3://")
        bucket, _, key = bucket_and_key.partition("/")
        return bucket, key

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

    # Store the graph
    graph_store.upsert("test_graph", G)

    # Get a summary of the graphs
    print("Graph Store Summary:")
    print(graph_store.summary())

    # Display detailed metadata
    print("Graph Store Details:")
    print(graph_store.details())

    # List all graphs
    print("List all graphs in the store:")
    print(graph_store.list())

    # Retrieve and display the graph
    retrieved_graph = graph_store.get("test_graph")
    print("Retrieved Graph Edges:", retrieved_graph.edges(data=True))

    # Check if the graph exists
    print("Graph exists:", graph_store.check("test_graph"))

    # Delete the graph
    graph_store.delete("test_graph")
    print("Graph exists after deletion:", graph_store.check("test_graph"))