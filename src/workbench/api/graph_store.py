"""GraphStore: Storage for NetworkX Graphs"""

import networkx as nx
from datetime import datetime
from typing import Union, Optional
import pandas as pd

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_graph_store import AWSGraphStore


class GraphStore(AWSGraphStore):
    """GraphStore: Storage for NetworkX Graphs

    Common Usage:
        ```python
        graph_store = GraphStore()

        # List all Graphs
        graph_store.list()

        # Add Graph
        g = nx.erdos_renyi_graph(5, 0.5)
        graph_store.upsert("/test/my_graph", my_graph)

        # Retrieve Graph
        g = graph_store.get("/test/my_graph")
        print(g)

        # Delete Graph
        graph_store.delete("/test/my_graph")
        ```
    """

    def __init__(self, path_prefix: Optional[str] = None):
        """GraphStore Init Method

        Args:
            path_prefix (Optional[str]): Path prefix for storage locations (Defaults to None)
        """

        # Initialize the SuperClass
        super().__init__(path_prefix=path_prefix)

    def summary(self) -> pd.DataFrame:
        """Provide a summary of all graphs in the store.

        Returns:
            pd.DataFrame: Summary DataFrame with location, size, and modified date.
        """
        return super().summary()

    def details(self) -> pd.DataFrame:
        """Return detailed metadata for all stored graphs.

        Returns:
            pd.DataFrame: DataFrame with details like location, size, and last modified date.
        """
        return super().details()

    def check(self, location: str) -> bool:
        """Check if a graph exists.

        Args:
            location (str): Logical location of the graph.

        Returns:
            bool: True if the graph exists, False otherwise.
        """
        return super().check(location)

    def get(self, location: str) -> Union[nx.Graph, None]:
        """Retrieve a NetworkX graph from AWS S3.

        Args:
            location (str): Logical location of the graph.

        Returns:
            Union[nx.Graph, None]: The retrieved graph or None if not found.
        """
        return super().get(location)

    def upsert(self, location: str, graph: nx.Graph):
        """Insert or update a NetworkX graph in AWS S3.

        Args:
            location (str): Logical location to store the graph.
            graph (nx.Graph): The NetworkX graph to store.
        """
        super().upsert(location, graph)

    def list(self) -> list:
        """List all graphs in the store.

        Returns:
            list: A list of all graph locations in the store.
        """
        return super().list()

    def last_modified(self, location: str) -> Union[datetime, None]:
        """Return the last modified date of a graph.

        Args:
            location (str): Logical location of the graph.

        Returns:
            Union[datetime, None]: Last modified datetime or None if not found.
        """
        return super().last_modified(location)

    def delete(self, location: str):
        """Delete a NetworkX graph from AWS S3.

        Args:
            location (str): Logical location of the graph to delete.
        """
        super().delete(location)


if __name__ == "__main__":
    """Exercise the GraphStore Class"""
    # Create an GraphStore instance
    graph_store = GraphStore()

    # Create a test graph
    G = nx.erdos_renyi_graph(5, 0.5)
    for u, v, d in G.edges(data=True):
        d["weight"] = 1.0

    # Store the graph
    graph_store.upsert("test/test_graph", G)

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
