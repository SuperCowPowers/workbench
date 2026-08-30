"""Workbench Local: Filesystem-backed artifacts for local model development.

These classes mirror the AWS artifact API against local storage, so a script
written locally publishes to AWS and produces the same model. Storage lives
under ``WORKBENCH_LOCAL_PATH`` (default ``~/.workbench/local``).

- LocalArtifact: Base class for all local artifacts
- DataSource: A DataFrame on local disk
- FeatureSet: Engineered features on local disk
- Model: A model trained on this machine by the generated model script
- Endpoint: In-process inference against a locally trained model
- Meta: Listings for the artifacts in local storage

PublicData is re-exported here: it reads public S3 anonymously, so it needs no
AWS account and is the usual starting point for a local model.
"""

from workbench.core.model_types import ModelType, ModelFramework
from workbench.public_data import PublicData

from .local_artifact import LocalArtifact
from .data_source import DataSource
from .feature_set import FeatureSet
from .model import Model
from .endpoint import Endpoint
from .meta import Meta

__all__ = [
    "LocalArtifact",
    "DataSource",
    "FeatureSet",
    "Model",
    "Endpoint",
    "Meta",
    "ModelType",
    "ModelFramework",
    "PublicData",
]
