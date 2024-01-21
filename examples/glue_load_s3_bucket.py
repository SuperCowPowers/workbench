import sys
import logging

# SageWorks Imports
import sageworks
from sageworks.api.data_source import DataSource
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.glue_utils import glue_args_to_dict, list_s3_files
from sageworks.utils.sageworks_logging import logging_setup

# Setup logging (note: regular prints don't show up in Glue Logs)
logging_setup(color_logs=False)
log = logging.getLogger("sageworks")
log.info(f"SageWorks: {sageworks.__version__}")

# Convert Glue Job Args to a Dictionary
glue_args = glue_args_to_dict(sys.argv)

# Set the SAGEWORKS_BUCKET for the ConfigManager
cm = ConfigManager()
cm.set_config("SAGEWORKS_BUCKET", glue_args["--sageworks-bucket"])
log.info(cm.get_all_config())

# List all the CSV files in the given S3 Path
input_s3_path = glue_args["--input-s3-path"]
for input_file in list_s3_files(input_s3_path):
    log.info(input_file)

    # Note: If we don't specify a name, one will be 'auto-generated'
    my_data = DataSource(input_file, name=None)
    log.info(f"DataSource {my_data.uuid} created from {input_file}")
