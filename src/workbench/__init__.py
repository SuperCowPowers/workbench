# Copyright (c) 2021-2024 SuperCowPowers LLC

"""
Workbench Main Classes
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
  - from workbench.core.transforms.data_loaders.light.json_to_data_source import JSONToDataSource
  - help(JSONToDataSource)


      class JSONToDataSource(workbench.core.transforms.transform.Transform)
     |  JSONToDataSource(json_file_path: str, data_name: str)
     |
     |  JSONToDataSource: Class to move local JSON Files into a Workbench DataSource
     |
     |  Common Usage:
     |      json_to_data = JSONToDataSource(json_file_path, data_name)
     |      json_to_data.set_output_tags(["abalone", "json", "whatever"])
     |      json_to_data.transform()
"""

import os
from importlib.metadata import version

try:
    __version__ = version("workbench")
except Exception:
    __version__ = "unknown"

# Workbench Logging
from workbench.utils.workbench_logging import logging_setup

# Check the environment variable to decide whether to set up logging
if os.getenv("WORKBENCH_SKIP_LOGGING", "False").lower() != "true":
    logging_setup()

# Route all awswrangler Athena queries through the Workbench workgroup. The
# workgroup carries a ResultConfiguration (S3 output location), so awswrangler
# does not fall back to the default results bucket. This global config is
# enforced across every wr.athena.* call in the process.
#
# Guarded so a minimal install (e.g. the workbench.lambda_layer subset) can
# import workbench without awswrangler/pandas present; full installs configure
# the workgroup as before.
try:
    import awswrangler as wr

    wr.config.workgroup = "workbench-workgroup"
except ImportError:
    # awswrangler is optional in minimal installs (e.g. the workbench.lambda_layer
    # subset); skip the global workgroup config. Importing workbench still succeeds.
    pass
