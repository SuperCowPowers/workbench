import sys

# SageWorks Imports
from sageworks.api.data_source import DataSource
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.glue_utils import get_resolved_options

# Convert Glue Job Args to a Dictionary
glue_args = get_resolved_options(sys.argv)

# Set the SAGEWORKS_BUCKET for the ConfigManager
cm = ConfigManager()
cm.set_config("SAGEWORKS_BUCKET", glue_args["sageworks-bucket"])

# Create a new Data Source from an S3 Path
source_path = "s3://sageworks-public-data/common/abalone.csv"
my_data = DataSource(source_path, name="abalone_glue_test")
