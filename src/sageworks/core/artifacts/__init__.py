"""Welcome to the SageWorks Core Artifacts Classes

These classes provide low-level APIs for interacting with the AWS services

- Artifact: Base class for all artifacts
- AthenaSource: Manages AWS Athena DataSources
- DataSourceAbstract: Abstract Class for defining DataSource Interfaces
- DataSourceFactory: A Factory Class that creates DataSource objects
- FeatureSetCore: Manages AWS Feature Store and Feature Groups
- ModelCore: Manages the training and deployment of AWS Model Groups and Packages
- EndpointCore: Manages the deployment and invocations/inference on AWS Endpoints
"""

from .artifact import Artifact
from .athena_source import AthenaSource
from .data_source_abstract import DataSourceAbstract
from .feature_set_core import FeatureSetCore
from .model_core import ModelCore
from .endpoint_core import EndpointCore

__all__ = ["Artifact", "AthenaSource", "DataSourceAbstract", "FeatureSetCore", "ModelCore", "EndpointCore"]
