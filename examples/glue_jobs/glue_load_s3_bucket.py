import sys

# Workbench Imports
from workbench.api.data_source import DataSource
from workbench.utils.config_manager import ConfigManager
from workbench.utils.glue_utils import get_resolved_options
from workbench.utils.aws_utils import list_s3_files

# Convert Glue Job Args to a Dictionary
glue_args = get_resolved_options(sys.argv)

# Set the WORKBENCH_BUCKET for the ConfigManager
cm = ConfigManager()
cm.set_config("WORKBENCH_BUCKET", glue_args["workbench-bucket"])

# List all the CSV files in the given S3 Path
input_s3_path = glue_args["input-s3-path"]
for input_file in list_s3_files(input_s3_path):
    # Note: If we don't specify a name, one will be 'auto-generated'
    my_data = DataSource(input_file, name=None)
