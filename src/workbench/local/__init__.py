"""Workbench Local: Filesystem-backed artifacts for local model development.

These classes mirror the AWS artifact API against local storage, so a script
written locally publishes to AWS and produces the same model. Storage lives
under ``WORKBENCH_LOCAL_PATH`` (default ``~/.workbench/local``).

- LocalArtifact: Base class for all local artifacts
- LocalDataSource: A DataFrame on local disk
- LocalFeatureSet: Engineered features on local disk
- LocalMeta: Listings for the artifacts in local storage
"""

from .local_artifact import LocalArtifact
from .local_data_source import LocalDataSource
from .local_feature_set import LocalFeatureSet
from .local_meta import LocalMeta

__all__ = ["LocalArtifact", "LocalDataSource", "LocalFeatureSet", "LocalMeta"]
