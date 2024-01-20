import sys
import logging

# SageWorks Imports
import sageworks
from sageworks.api.data_source import DataSource
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.glue_utils import glue_args_to_dict
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

# Create a new Data Source from an S3 Path
source_path = "s3://sageworks-public-data/common/abalone.csv"
my_data = DataSource(source_path, name="abalone_glue_test")
log.info(my_data.summary())
log.info(my_data.details())
