"""This Script creates the Graph Workbench Artifacts in AWS needed for the tests

Graphs:
    - test/karate_club
"""

import sys
import logging
from pathlib import Path

# Workbench Imports
from workbench.api.graph_store import GraphStore
from workbench.utils import graph_utils

# Setup the logger
log = logging.getLogger("workbench")

if __name__ == "__main__":

    # Get our GraphStore
    graph_store = GraphStore()

    # Get the path to the dataset in the repository data directory
    karate_graph = Path(sys.modules["workbench"].__file__).parent.parent.parent / "data" / "karate_graph.json"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the karate_club Graph
    if recreate or graph_store.check("test/karate_club") is False:
        karate_graph = graph_utils.load_graph_from_file(karate_graph)
        karate_graph.name = "karate_club"
        graph_store.upsert("test/karate_club", karate_graph)
        log.info("Created karate_club graph")
