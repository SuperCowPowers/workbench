import os
import networkx as nx
import boto3
import json
from datetime import datetime
from pathlib import Path

# Sageworks imports
from sageworks.core.artifacts.artifact import Artifact


class GraphCore(Artifact):
    """GraphCore: A class to handle graph artifacts in SageWorks"""

    def __init__(self, uuid: str):
        """Initialize the GraphCore class

        Args:
            uuid (str): The UUID of this graph artifact
        """
        super().__init__(uuid)
        self.s3_client = boto3.client('s3')
        self.graph = None  # Placeholder for the NetworkX graph object

        # Attempt to load the graph from S3
        if self.exists():
            self.load_graph()
        else:
            self.log.warning(f"Graph {self.uuid} does not exist in S3.")

    def __init__(self, source: str, name: str = None, tags: list = None):
        """
        Initializes a new GraphCore object.

        Args:
            source (str): The source of the graph. This can be an S3 path, file path, or an existing Graph object.
            name (str): The name of the graph (must be lowercase). If not specified, a name will be generated.
            tags (list[str]): A list of tags associated with the graph. If not specified, tags will be generated.
        """

        # Check the name
        if name is None:
            name = Artifact.generate_valid_name(source)
        else:
            Artifact.ensure_valid_name(name)

        # Base class initialization
        super().__init__(name)

        # Grab our S3 client
        self.s3_client = self.boto_session.client("s3")

        # Convert PosixPath to string if necessary
        if isinstance(source, Path):
            source = str(source)

        # Check if the source is an existing SageWorks graph, a NetworkX graph, a S3 path, or a file path
        if self.exists():
            self.graph = self.load_graph()
        elif isinstance(source, nx.Graph):
            self.graph = source
            self.graph.name = name
            self.save_graph(self.graph)
        elif source.startswith("s3://"):
            self.graph = self.load_graph(source)
        else:
            # Check if the source is a file path
            if os.path.exists(source):
                self._load_graph_from_file(source)
                if self.graph:
                    self.graph.name = name
                    self.save_graph(self.graph)
            else:
                self.log.warning(f"Could not find graph: {source}")
                self.graph = None
                return

        # Set the tags
        tags = [name] if tags is None else tags
        # FIXME: self.set_tags(tags)

    def exists(self) -> bool:
        """Check if the graph exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.sageworks_bucket, Key=f"graphs/{self.uuid}.json")
            return True
        except self.s3_client.exceptions.ClientError:
            return False

    def refresh_meta(self):
        """Refresh the metadata for the graph"""
        # TBD implementation to refresh metadata
        pass

    def onboard(self) -> bool:
        """Onboard this graph into SageWorks"""
        if not self.exists():
            self.log.info(f"Graph {self.uuid} does not exist, cannot onboard.")
            return False
        # TBD implementation for onboarding process
        self.set_status("onboarded")
        return True

    def details(self) -> dict:
        """Additional details about this graph"""
        if not self.graph:
            return {}
        return {
            "name": self.graph.name,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph)
        }

    def size(self) -> float:
        """Return the size of this graph artifact in MegaBytes"""
        if not self.graph:
            return 0.0
        graph_str = json.dumps(nx.readwrite.json_graph.node_link_data(self.graph))
        return len(graph_str.encode('utf-8')) / (1024 * 1024)

    def created(self) -> datetime:
        """Return the datetime when this graph artifact was created"""
        return datetime.now()  # TBD implementation

    def modified(self) -> datetime:
        """Return the datetime when this graph artifact was last modified"""
        return datetime.now()  # TBD implementation

    def arn(self):
        """AWS ARN (Amazon Resource Name) for this graph artifact"""
        return f"arn:aws:s3:::{self.sageworks_bucket}/graphs/{self.uuid}.json"

    def aws_url(self):
        """AWS console/web interface for this graph artifact"""
        return f"https://s3.console.aws.amazon.com/s3/object/{self.sageworks_bucket}?prefix=graphs/{self.uuid}.json"

    def aws_meta(self) -> dict:
        """Get the full AWS metadata for this graph artifact"""
        # TBD implementation
        return {}

    def delete(self):
        """Delete this graph artifact including all related AWS objects"""
        self.s3_client.delete_object(Bucket=self.sageworks_bucket, Key=f"graphs/{self.uuid}.json")
        self.log.info(f"Graph {self.uuid} deleted from S3")

    def save_graph(self, graph: nx.Graph) -> None:
        """Save the graph to S3

        Args:
            graph (nx.Graph): The NetworkX graph to save
        """
        self.graph = graph
        graph_json = nx.readwrite.json_graph.node_link_data(graph)
        graph_str = json.dumps(graph_json)
        self.s3_client.put_object(Bucket=self.sageworks_bucket, Key=f"graphs/{self.uuid}.json", Body=graph_str)

    def load_graph(self, s3_path: str = None) -> nx.Graph:
        """Load the graph from S3"""

        # Is the S3 path provided?
        if s3_path:
            # Split into bucket and key
            bucket, key = s3_path.split("/", 1)
        else:
            bucket = self.sageworks_bucket
            key = f"graphs/{self.uuid}.json"

        # Load the graph from S3
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        graph_str = response['Body'].read().decode('utf-8')
        graph_json = json.loads(graph_str)
        self.graph = nx.readwrite.json_graph.node_link_graph(graph_json)
        return self.graph

    def _load_graph_from_file(self, file_path: str):
        """Helper method to load the graph from a file path"""
        try:
            with open(file_path, 'r') as file:
                graph_json = json.load(file)
            self.graph = nx.readwrite.json_graph.node_link_graph(graph_json)
        except FileNotFoundError:
            self.log.error(f"File not found: {file_path}")
            self.graph = None
        except json.JSONDecodeError:
            self.log.error(f"Error decoding JSON from file: {file_path}")
            self.graph = None
        except Exception as e:
            self.log.error(f"An error occurred while loading the graph from file: {file_path}. Error: {e}")
            self.graph = None


# Example usage
if __name__ == "__main__":

    # Create a GraphCore object
    graph = GraphCore("karate_club")

    # Print the details
    print(graph.details())
