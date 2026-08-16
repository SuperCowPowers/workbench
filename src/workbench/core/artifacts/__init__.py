"""Welcome to the Workbench Core Artifacts Classes

These classes provide low-level APIs for interacting with the AWS services

- Artifact: Storage-agnostic base class for all artifacts
- AWSArtifact: Base class for AWS-backed artifacts
- AthenaSource: Manages AWS Athena DataSources
- DataSourceAbstract: Abstract Class for defining DataSource Interfaces
- DataSourceFactory: A Factory Class that creates DataSource objects
- FeatureSetCore: Manages AWS Feature Store and Feature Groups
- ModelCore: Manages the training and deployment of AWS Model Groups and Packages
- EndpointCore: Manages the deployment and invocations/inference on AWS Endpoints
"""

from workbench.core.artifact import Artifact
from .aws_artifact import AWSArtifact
from .athena_source import AthenaSource
from .data_source_abstract import DataSourceAbstract
from .feature_set_core import FeatureSetCore
from .model_core import ModelCore, ModelType, ModelFramework
from .endpoint_core import EndpointCore

__all__ = [
    "Artifact",
    "AWSArtifact",
    "AthenaSource",
    "DataSourceAbstract",
    "FeatureSetCore",
    "ModelCore",
    "ModelType",
    "ModelFramework",
    "EndpointCore",
]
