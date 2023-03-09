"""Transform: Abstract Base Class for all transforms in SageWorks"""
from abc import ABC, abstractmethod
from enum import Enum, auto


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
    def __init__(self):
        """Transform: Abstract Base Class for all transforms in SageWorks"""

    @abstractmethod
    def input_type(self) -> TransformInput:
        """What Input Type does this Transform Consume"""
        pass

    @abstractmethod
    def output_type(self) -> TransformOutput:
        """What Output Type does this Transform Produce"""
        pass

    @abstractmethod
    def set_input_uuid(self, input_uuid: str):
        """Set the Input UUID (Name) for this Transform"""
        pass

    @abstractmethod
    def set_output_uuid(self, output_uuid: str):
        """Set the Output UUID (Name) for this Transform"""
        pass

    @abstractmethod
    def transform(self, overwrite: bool = True):
        """Perform the Transformation from Input to Output
           Args:
               overwrite (bool): Overwrite the output/uuid if it exists (default = True)
        """
        pass

    @abstractmethod
    def get_output(self) -> any:
        """Get the Output from this Transform"""
        pass

    @abstractmethod
    def validate_input(self) -> bool:
        """Validate the Input for this Transform"""
        pass
