"""Workbench Local: Filesystem-backed artifacts for local model development.

These classes mirror the AWS artifact API against local storage, so a script
written locally publishes to AWS and produces the same model. Storage lives
under ``WORKBENCH_LOCAL_PATH`` (default ``~/.workbench/local``).

- LocalArtifact: Base class for all local artifacts
- LocalDataSource: A DataFrame on local disk
- LocalFeatureSet: Engineered features on local disk
- LocalModel: A model trained on this machine by the generated model script
- LocalEndpoint: In-process inference against a locally trained model
- LocalMeta: Listings for the artifacts in local storage

PublicData is re-exported here: it reads public S3 anonymously, so it needs no
AWS account and is the usual starting point for a local model.
"""

from workbench.core.model_types import ModelType, ModelFramework
from workbench.public_data import PublicData

from .local_artifact import LocalArtifact
from .local_data_source import LocalDataSource
from .local_feature_set import LocalFeatureSet
from .local_model import LocalModel
from .local_endpoint import LocalEndpoint
from .local_meta import LocalMeta

__all__ = [
    "LocalArtifact",
    "LocalDataSource",
    "LocalFeatureSet",
    "LocalModel",
    "LocalEndpoint",
    "LocalMeta",
    "ModelType",
    "ModelFramework",
    "PublicData",
]
