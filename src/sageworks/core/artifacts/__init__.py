"""Welcome to the SageWorks API Classes

These class provide high-level APIs for the SageWorks package, offering easy access to its core classes:

- DataSource: Manages AWS Data Catalog and Athena
- FeatureSet: Manages AWS Feature Store and Feature Groups
- Model: Manages the training and deployment of AWS Model Groups and Packages
- Endpoint: Manages the deployment and invocations/inference on AWS Endpoints
"""
from .artifact import Artifact
from .athena_source import AthenaSource
from .data_source_abstract import DataSourceAbstract
from .feature_set_core import FeatureSetCore
from .model_core import ModelCore
from .endpoint_core import EndpointCore

__all__ = ["Artifact", "AthenaSource", "DataSourceAbstract", "FeatureSetCore", "ModelCore", "EndpointCore"]
