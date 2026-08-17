"""Enumerated types for Workbench Models.

These live outside the artifact classes so script generation and local model
development can use them without pulling in an AWS session.
"""

from enum import Enum


class ModelType(Enum):
    """Enumerated Types for Workbench Model Types"""

    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    CLUSTERER = "clusterer"
    PROXIMITY = "proximity"
    PROJECTION = "projection"
    UQ_REGRESSOR = "uq_regressor"
    ENSEMBLE_REGRESSOR = "ensemble_regressor"
    TRANSFORMER = "transformer"
    UNKNOWN = "unknown"


class ModelFramework(Enum):
    """Enumerated Types for Workbench Model Frameworks"""

    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    PYTORCH = "pytorch"
    CHEMPROP = "chemprop"
    TRANSFORMER = "transformer"
    META = "meta"
    UNKNOWN = "unknown"
