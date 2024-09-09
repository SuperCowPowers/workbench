import sys

# SageWorks Imports
from sageworks.api.data_source import DataSource
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.glue_utils import get_resolved_options
from sageworks.utils.aws_utils import list_s3_files

# Convert Glue Job Args to a Dictionary
glue_args = get_resolved_options(sys.argv)

# Set the SAGEWORKS_BUCKET for the ConfigManager
cm = ConfigManager()
cm.set_config("SAGEWORKS_BUCKET", glue_args["sageworks-bucket"])

# List all the CSV files in the given S3 Path
input_s3_path = glue_args["input-s3-path"]
for input_file in list_s3_files(input_s3_path):
    # Note: If we don't specify a name, one will be 'auto-generated'
    my_data = DataSource(input_file, name=None)
