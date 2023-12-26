# Copyright (c) 2021-2024 SuperCowPowers LLC

"""
SageWorks Main Classes
- Artifacts
  - DataSource
  - FeatureSet
  - Model
  - Endpoint
- Transforms
  - DataLoaders
  - DataToData
  - DataToFeatures
  - FeaturesToModel
  - ModelToEndpoint

  For help on particular classes you can do this
  - from sageworks.core.transforms.data_loaders.light.json_to_data_source import JSONToDataSource
  - help(JSONToDataSource)


      class JSONToDataSource(sageworks.core.transforms.transform.Transform)
     |  JSONToDataSource(json_file_path: str, data_uuid: str)
     |
     |  JSONToDataSource: Class to move local JSON Files into a SageWorks DataSource
     |
     |  Common Usage:
     |      json_to_data = JSONToDataSource(json_file_path, data_uuid)
     |      json_to_data.set_output_tags(["abalone", "json", "whatever"])
     |      json_to_data.transform()
"""
from importlib.metadata import version

try:
    __version__ = version("sageworks")
except Exception:
    __version__ = "unknown"

# SageWorks Logging
from sageworks.utils.sageworks_logging import logging_setup

logging_setup()
