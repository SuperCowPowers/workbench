"""This Script creates the Graph SageWorks Artifacts in AWS needed for the tests

Graphs:
    - karate_club
"""

import sys
import logging
from pathlib import Path
from sageworks.api.data_source import DataSource
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint
from sageworks.core.artifacts.graph_core import GraphCore

from sageworks.utils.test_data_generator import TestDataGenerator
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker

# Setup the logger
log = logging.getLogger("sageworks")

if __name__ == "__main__":
    # This forces a refresh on all the data we get from the AWS Broker
    AWSServiceBroker().get_all_metadata(force_refresh=True)

    # Get the path to the dataset in the repository data directory
    karate_graph = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "karate_graph.json"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the karate_club Graph
    if recreate or not GraphCore("karate_club").exists():
        GraphCore(karate_graph, name="karate_club")
        log.info("Created karate_club graph")
