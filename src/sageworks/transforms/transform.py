"""Transform: Abstract Base Class for all transforms in SageWorks"""
from abc import ABC, abstractmethod
from enum import Enum


class TransformInput(Enum):
    """Enumerated Types for SageWorks Transform Inputs"""
    S3_OBJECT = 0
    DATA_SOURCE = 1
    FEATURE_SET = 2
    MODEL = 3


class TransformOutput(Enum):
    """Enumerated Types for SageWorks Transform Outputs"""
    DATA_SOURCE = 1
    FEATURE_SET = 2
    MODEL = 3
    ENDPOINT = 4


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
    def set_input(self, resource_url: str):
        """Set the Input for this Transform"""
        pass

    @abstractmethod
    def set_output_name(self, uuid: str):
        """Set the Output Name (uuid) for this Transform"""
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

    @abstractmethod
    def validate_output_pre_transform(self) -> bool:
        """Validate, output type, AWS write permissions, etc. before it's created"""
        pass

    @abstractmethod
    def validate_output(self) -> bool:
        """Validate the Output after it's been created"""
        pass
