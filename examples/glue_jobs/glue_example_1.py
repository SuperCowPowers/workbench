# Example Glue Job that simply prints the first few rows of a DataSource
import sys

# Workbench Imports
from workbench.utils.config_manager import ConfigManager
from workbench.utils.glue_utils import get_resolved_options

# Convert Glue Job Args to a Dictionary
glue_args = get_resolved_options(sys.argv)

# Set the WORKBENCH_BUCKET for the ConfigManager
cm = ConfigManager()
cm.set_config("WORKBENCH_BUCKET", glue_args["workbench-bucket"])

# Important Note: This import needs to happen after the WORKBENCH_BUCKET is set
from workbench.api import DataSource  # noqa: E402

# Grab a test DataSource
ds = DataSource("abalone_data")
df = ds.pull_dataframe()
print(df.head())
