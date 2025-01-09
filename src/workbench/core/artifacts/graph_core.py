import os
import json
from datetime import datetime
from pathlib import Path
from typing import Union

# NetworkX import
try:
    import networkx as nx
except ImportError:
    print("NetworkX Python module not found! pip install networkx")
    raise ImportError("NetworkX Python module not found! pip install networkx")

# Workbench imports
from workbench.core.artifacts.artifact import Artifact
from workbench.utils.json_utils import CustomEncoder


class GraphCore(Artifact):
    """GraphCore: A class to handle graph artifacts in Workbench"""

    def __init__(self, source: Union[str, nx.Graph], name: str = None):
        """
        Initializes a new GraphCore object.

        Args:
            source (str): The source of the graph. S3 path, file path, nx_graph, or an existing Graph object.
            name (str): The name of the graph (must be lowercase). If not specified, a name will be generated.
        """

        # Check the name
        if name is None:
            if isinstance(source, str):
                name = Artifact.generate_valid_name(source)
            elif isinstance(source, nx.Graph):
                name = source.name

        # Ensure the name is valid
        Artifact.is_name_valid(name, lower_case=False)

        # Call our parent class constructor
        super().__init__(name)

        # Grab our S3 client
        self.s3_client = self.boto3_session.client("s3")

        # Convert PosixPath to string if necessary
        if isinstance(source, Path):
            source = str(source)

        # Check if the source is a NetworkX graph, a S3 path, file path, or an existing Workbench graph
        if isinstance(source, nx.Graph):
            self.graph = source
            self.graph.name = name if name else self.graph.name
            self.save()

        # Check if we have an S3 path
        elif source.startswith("s3://"):
            self.graph = self.load_graph(source)

        # Check if the source is a file path
        elif os.path.exists(source):
            self._load_graph_from_file(source)
            if self.graph:
                self.graph.name = name
                self.save()

        # Check if the source is an existing Workbench graph
        elif self.exists():
            self.graph = self.load_graph()

        # Otherwise, we could not find the graph
        else:
            self.log.warning(f"Could not find graph: {source}")
            self.graph = None
            return

        # Tags can be set with the set_tags method
        self.tags = None

    def exists(self) -> bool:
        """Check if the graph exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.workbench_bucket, Key=f"graphs/{self.uuid}.json")
            return True
        except self.s3_client.exceptions.ClientError:
            return False

    def get_tags(self, tag_type: str = "user") -> list:
        """Get the tags for this graph"""
        if tag_type == "user":
            return self.tags
        else:
            return []

    def refresh_meta(self):
        """Refresh the metadata for the graph"""
        self.workbench_meta()

    def workbench_meta(self) -> dict:
        """Get the Workbench specific metadata for this Graph"""
        return {"tags": self.tags}

    def onboard(self) -> bool:
        """Onboard this graph into Workbench"""
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
            "description": self.graph.name,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
        }

    def size(self) -> float:
        """Return the size of this graph artifact in MegaBytes"""
        if not self.graph:
            return 0.0
        graph_str = json.dumps(nx.readwrite.json_graph.node_link_data(self.graph))
        return len(graph_str.encode("utf-8")) / (1024 * 1024)

    def created(self) -> datetime:
        """Get the creation date for this graph artifact"""
        # Note: Since S3 does not store creation date, we will use the last modified date
        return self.modified()

    def modified(self) -> datetime:
        """Get the last modified date for this graph artifact"""
        response = self.s3_client.head_object(Bucket=self.workbench_bucket, Key=f"graphs/{self.uuid}.json")
        return response["LastModified"]

    def hash(self) -> str:
        """Get the hash for this graph artifact"""
        return "TBD"

    def arn(self):
        """AWS ARN (Amazon Resource Name) for this graph artifact"""
        return f"arn:aws:s3:::{self.workbench_bucket}/graphs/{self.uuid}.json"

    def aws_url(self):
        """AWS console/web interface for this graph artifact"""
        return f"https://s3.console.aws.amazon.com/s3/object/{self.workbench_bucket}?prefix=graphs/{self.uuid}.json"

    def aws_meta(self) -> dict:
        """Get the full AWS metadata for this graph artifact"""
        return {}

    def delete(self):
        """Delete this graph artifact including all related AWS objects"""
        self.s3_client.delete_object(Bucket=self.workbench_bucket, Key=f"graphs/{self.uuid}.json")
        self.log.info(f"Graph {self.uuid} deleted from S3")

    def get_nx_graph(self) -> nx.Graph:
        """Return the NetworkX graph"""
        return self.graph

    def set_nx_graph(self, graph: nx.Graph) -> None:
        """Set the NetworkX graph"""
        self.graph = graph

    def save(self) -> None:
        """Save the internal NetworkX graph to S3"""
        graph_json = nx.readwrite.json_graph.node_link_data(self.graph)
        graph_str = json.dumps(graph_json, cls=CustomEncoder)
        self.s3_client.put_object(Bucket=self.workbench_bucket, Key=f"graphs/{self.uuid}.json", Body=graph_str)

    def graph_layout(self, layout: str = "spring") -> dict:
        """Compute the layout of the graph using the specified algorithm"""
        if layout == "spring":
            pos = nx.spring_layout(self.graph, iterations=500)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout == "spectral":
            pos = nx.spectral_layout(self.graph)
        elif layout == "shell":
            pos = nx.shell_layout(self.graph)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph, iterations=500)

        # Now store the positions in the graph and save the graph
        self.store_node_positions(pos)
        self.save()

    def store_node_properties(self, node_properties: dict) -> None:
        """Store node properties as attributes in the NetworkX graph"""
        for node, properties in node_properties.items():
            self.graph.nodes[node].update(properties)

    def store_node_positions(self, node_positions: dict) -> None:
        """Store node positions in the NetworkX graph"""
        for node, coords in node_positions.items():
            self.graph.nodes[node]["pos"] = list(coords)

    def load_graph(self, s3_path: str = None) -> nx.Graph:
        """Load the NetworkX graph from S3"""

        # Is the S3 path provided?
        if s3_path:
            # Split into bucket and key
            bucket, key = s3_path.split("/", 1)
        else:
            bucket = self.workbench_bucket
            key = f"graphs/{self.uuid}.json"

        # Load the graph from S3
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        graph_str = response["Body"].read().decode("utf-8")
        graph_json = json.loads(graph_str)
        self.graph = nx.readwrite.json_graph.node_link_graph(graph_json)
        return self.graph

    def _load_graph_from_file(self, file_path: str):
        """Helper method to load the graph from a file path"""
        try:
            with open(file_path, "r") as file:
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
    from workbench.web_interface.components.plugins.graph_plot import GraphPlot
    from workbench.core.artifacts.graph_core import GraphCore  # noqa
    from workbench.utils.graph_utils import create_nxgraph_from_dfs
    import pandas as pd

    # Create an NetworkX Graph from two dataframes
    node_df = pd.DataFrame({"node_id": [1, 2, 3], "attribute": ["A", "B", "C"]})
    edges_df = pd.DataFrame({"source": [1, 2], "target": [2, 3], "weight": [0.5, 1.5]})
    nx_graph = create_nxgraph_from_dfs(node_df, edges_df)
    nx_graph.name = "my_graph"
    my_graph = GraphCore(nx_graph)

    # Grab an existing graph
    my_graph = GraphCore("karate_club")

    # Print the details
    print(my_graph.details())

    # Layout the graph using the spring algorithm
    my_graph.graph_layout(layout="spring")

    # Note: You can set the node positions which allows you to use any layout algorithm
    # pos = nx.spring_layout(graph.get_nx_graph())
    # graph.store_node_positions(pos)

    # View the graph
    graph_plot = GraphPlot()
    [figure, *_] = graph_plot.update_properties(my_graph)
    figure.show()
