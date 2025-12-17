# Example Glue Job that runs inference on a particular model
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
from workbench.api import Endpoint  # noqa: E402

# Run Inference on a particular model (using its endpoint)
end = Endpoint("abalone-regression")
df = end.auto_inference()
print(df.columns)
print(df.head())
