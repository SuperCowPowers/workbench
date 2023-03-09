"""Transform: Base Class for all transforms within SageWorks
              Inherited Classes must implement the abstract transform() method"""
from abc import ABC, abstractmethod
from enum import Enum, auto
import logging

from sageworks.utils.logging import logging_setup

# Setup Logging
logging_setup()


class TransformInput(Enum):
    """Enumerated Types for SageWorks Transform Inputs"""
    PANDAS_DF = auto()
    SPARK_DF = auto()
    S3_OBJECT = auto()
    DATA_SOURCE = auto()
    FEATURE_SET = auto()
    MODEL = auto()


class TransformOutput(Enum):
    """Enumerated Types for SageWorks Transform Outputs"""
    PANDAS_DF = auto()
    SPARK_DF = auto()
    S3_OBJECT = auto()
    DATA_SOURCE = auto()
    FEATURE_SET = auto()
    MODEL = auto()
    ENDPOINT = auto()


class Transform(ABC):
    def __init__(self, input_uuid: str = None, output_uuid: str = None):
        """Transform: Abstract Base Class for all transforms in SageWorks"""

        # FIXME: Should this class be a Python dataclass?
        self.log = logging.getLogger(__name__)
        self.input_type = None
        self.output_type = None
        self.input_uuid = input_uuid
        self.output_uuid = output_uuid

        # FIXME: We should have this come from AWS or Config
        self.data_catalog_db = 'sageworks'
        self.data_source_s3_path = 's3://sageworks-data-sources'
        self.feature_sets_s3_path = 's3://sageworks-feature-sets'

    @abstractmethod
    def transform(self, overwrite: bool = True):
        """Perform the Transformation from Input to Output
           Args:
               overwrite (bool): Overwrite the output_uuid if it exists (default = True)
        """
        pass

    def input_type(self) -> TransformInput:
        """What Input Type does this Transform Consume"""
        return self.input_type

    def output_type(self) -> TransformOutput:
        """What Output Type does this Transform Produce"""
        return self.output_type

    def set_input_uuid(self, input_uuid: str):
        """Set the Input UUID (Name) for this Transform"""
        self.input_uuid = input_uuid

    def set_output_uuid(self, output_uuid: str):
        """Set the Output UUID (Name) for this Transform"""
        self.output_uuid = output_uuid
