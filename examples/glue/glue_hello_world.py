import sys

# Workbench Imports
from workbench.api.data_source import DataSource
from workbench.utils.config_manager import ConfigManager
from workbench.utils.glue_utils import get_resolved_options

# Convert Glue Job Args to a Dictionary
glue_args = get_resolved_options(sys.argv)

# Set the WORKBENCH_BUCKET for the ConfigManager
cm = ConfigManager()
cm.set_config("WORKBENCH_BUCKET", glue_args["workbench-bucket"])

# Create a new Data Source from an S3 Path
source_path = "s3://workbench-public-data/common/abalone.csv"
my_data = DataSource(source_path, name="abalone_glue_test")
