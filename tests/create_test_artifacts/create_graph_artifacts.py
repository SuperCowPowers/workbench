"""This Script creates the Graph SageWorks Artifacts in AWS needed for the tests

Graphs:
    - karate_club
"""

import sys
import logging
from pathlib import Path
from sageworks.core.artifacts.graph_core import GraphCore

# Setup the logger
log = logging.getLogger("sageworks")

if __name__ == "__main__":

    # Get the path to the dataset in the repository data directory
    karate_graph = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "karate_graph.json"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the karate_club Graph
    if recreate or not GraphCore("karate_club").exists():
        GraphCore(karate_graph, name="karate_club")
        log.info("Created karate_club graph")
