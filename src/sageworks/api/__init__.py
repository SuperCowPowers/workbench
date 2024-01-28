"""Welcome to the SageWorks API Classes

These class provide high-level APIs for the SageWorks package, offering easy access to its core classes:

- DataSource: Manages AWS Data Catalog and Athena
- FeatureSet: Manages AWS Feature Store and Feature Groups
- Model: Manages the training and deployment of AWS Model Groups and Packages
- Endpoint: Manages the deployment and invocations/inference on AWS Endpoints
"""

from .data_source import DataSource
from .feature_set import FeatureSet
from .model import Model
from .endpoint import Endpoint

__all__ = ["DataSource", "FeatureSet", "Model", "Endpoint"]
