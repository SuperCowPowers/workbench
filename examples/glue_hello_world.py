import sys

# SageWorks Imports
from sageworks.api.data_source import DataSource
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.glue_utils import glue_args_to_dict


# Convert Glue Job Args to a Dictionary
glue_args = glue_args_to_dict(sys.argv)

# Set the SAGEWORKS_BUCKET for the ConfigManager
cm = ConfigManager()
cm.set_config("SAGEWORKS_BUCKET", glue_args["--sageworks_bucket"])
print(cm.get_all_config())

# Create a new Data Source from an S3 Path (or a local file)
# source_path = "s3://sageworks-public-data/common/abalone.csv"
# my_data = DataSource(source_path, name="abalone_glue_test")
# log.info(my_data.summary())
# log.info(my_data.details())

ds = DataSource("abalone_data")
print(ds.details())
