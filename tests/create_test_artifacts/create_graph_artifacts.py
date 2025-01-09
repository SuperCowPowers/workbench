"""This Script creates the Graph Workbench Artifacts in AWS needed for the tests

Graphs:
    - karate_club
"""

import sys
import logging
from pathlib import Path
from workbench.utils import graph_utils

# Setup the logger
log = logging.getLogger("workbench")

if __name__ == "__main__":

    # Get the path to the dataset in the repository data directory
    karate_graph = Path(sys.modules["workbench"].__file__).parent.parent.parent / "data" / "karate_graph.json"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the karate_club Graph
    if recreate or not graph_utils.exists("karate_club"):
        karate_graph = graph_utils.load_graph_from_file(karate_graph)
        karate_graph.name = "karate_club"
        graph_utils.save(karate_graph)
        log.info("Created karate_club graph")
