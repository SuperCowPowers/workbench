!!! tip inline end "AWS Glue Simplified"
    AWS Glue Jobs are a great way to automate ETL and data processing. SageWorks takes all the hassle out of creating and debugging Glue Jobs. Follow this guide and empower your Glue Jobs with SageWorks!

SageWorks make creating, testing, and debugging of AWS Glue Jobs easy. The exact same [SageWorks API Classes](../api_classes/overview.md) are used in your Glue Jobs. Also since SageWorks manages the roles for both API and Glue Jobs you'll be able to test new Glue Jobs locally and minimizes surprises when deploying your Glue Job.

## Glue Job Setup

Setting up a AWS Glue Job that uses SageWorks is straight forward. SageWorks can be 'installed' on AWS Glue via the `--additional-python-modules` parameter and then you can use the Sageworks API just like normal. 

<img alt="sageworks_repl" style="float: right; width: 348px; padding-left: 10px; border: 1px solid grey;""
src="https://github.com/SuperCowPowers/sageworks/assets/4806709/64b72d12-e5d6-411a-9ce5-a64926afceea">

Here are the settings and a screen shot to guide you. Please feel free to contact SageWorks support if you need any help with setting up Glue Jobs.

- IAM Role: SageWorks-ExecutionRole
- Type: Spark
- Glue Version: Glue 4.0
- Worker Type: G.1X
- Number of Workers: 2
- Job Parameters
  - --additional-python-modules: sageworks>=0.4.5
  - --sageworks-bucket: <your bucket\>

## SageWorks Glue Example

```py title="examples/glue_hello_world.py"
import sys

# SageWorks Imports
from sageworks.api.data_source import DataSource
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.glue_utils import glue_args_to_dict

# Convert Glue Job Args to a Dictionary
glue_args = glue_args_to_dict(sys.argv)

# Set the SAGEWORKS_BUCKET for the ConfigManager
cm = ConfigManager()
cm.set_config("SAGEWORKS_BUCKET", glue_args["--sageworks-bucket"])

# Create a new Data Source from an S3 Path
source_path = "s3://sageworks-public-data/common/abalone.csv"
my_data = DataSource(source_path, name="abalone_glue_test")
```

## Glue Job Local Testing
Glue Power without the Pain. SageWorks manages the AWS Execution Role, so local API and Glue Jobs will have the same permissions/access. Also using the same Code as your notebooks or scripts makes creating and testing Glue Jobs a breeze.

```shell
export SAGEWORKS_CONFIG=<your config>  # Only if not already set up
python my_glue_job.py --sageworks-bucket <your bucket>
```

## Additional Resources
- SageWorks Glue Jobs: [SageWorks Glue](https://docs.google.com/presentation/d/1Bdcve27BDLbUkslZJAc1OrG6VkDopEtnL8Wh8HaLrco/edit?usp=sharing)
- Setting up SageWorks on your AWS Account: [AWS Setup](../aws_setup/core_stack.md)
- Using SageWorks for ML Pipelines: [SageWorks API Classes](../api_classes/overview.md)

<img align="right" src="../images/scp.png" width="180">

- SageWorks Core Classes: [Core Classes](../core_classes/overview.md)
- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
