!!! tip inline end "AWS Glue Simplified"
    AWS Glue Jobs are a great way to automate ETL and data processing. Workbench takes all the hassle out of creating and debugging Glue Jobs. Follow this guide and empower your Glue Jobs with Workbench!

Workbench make creating, testing, and debugging of AWS Glue Jobs easy. The exact same [Workbench API Classes](../api_classes/overview.md) are used in your Glue Jobs. Also since Workbench manages the roles for both API and Glue Jobs you'll be able to test new Glue Jobs locally and minimizes surprises when deploying your Glue Job.

## Glue Job Setup

Setting up a AWS Glue Job that uses Workbench is straight forward. Workbench can be 'installed' on AWS Glue via the `--additional-python-modules` parameter and then you can use the Workbench API just like normal. 

<img alt="workbench_repl" style="float: right; width: 348px; padding-left: 12px; border: 1px solid grey;""
src="https://github.com/SuperCowPowers/workbench/assets/4806709/64b72d12-e5d6-411a-9ce5-a64926afceea">

Here are the settings and a screen shot to guide you. There are several ways to set up and run Glue Jobs, with either the Workbench-ExecutionRole or using the WorkbenchAPIPolicy. Please feel free to contact Workbench support if you need any help with setting up Glue Jobs.

- IAM Role: Workbench-ExecutionRole
- Type: Spark
- Glue Version: Glue 4.0
- Worker Type: G.1X
- Number of Workers: 2
- Job Parameters
  - --additional-python-modules: workbench>=0.4.6
  - --workbench-bucket: <your workbench bucket\>

!!! tip "Glue IAM Role Details"
    If your Glue Jobs already use an existing IAM Role then you can add the `WorkbenchAPIPolicy` to that Role to enable the Glue Job to perform Workbench API Tasks.

## Workbench Glue Example
Anyone familiar with a typical Glue Job should be pleasantly surpised by how simple the example below is. Also Workbench allows you to test Glue Jobs locally using the same code that you use for script and Notebooks (see [Glue Testing](#glue-job-local-testing))
!!!tip "Glue Job Arguments"
    AWS Glue Jobs take arguments in the form of **Job Parameters** (see screenshot above). There's a Workbench utility function `get_resolved_options` that turns these Job Parameters into a nice dictionary for ease of use.

```py title="examples/glue_hello_world.py"
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
```

## Glue Example 2
This example takes two 'Job Parameters'

- --workbench-bucket : <your workbench bucket\>
- --input-s3-path : <your S3 input path\>

The example will convert all CSV files in an S3 bucket/prefix and load them up as DataSources in Workbench.

```py title="examples/glue_load_s3_bucket.py"
import sys

# Workbench Imports
from workbench.api.data_source import DataSource
from workbench.utils.config_manager import ConfigManager
from workbench.utils.glue_utils import get_resolved_options, list_s3_files

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
```

## Exception Log Forwarding
When a Glue Job crashes (has an exception), the AWS console will show you the last line of the exception, this is mostly useless. If you use Workbench log forwarding the exception/stack will be forwarded to CloudWatch.

```py
from workbench.utils.workbench_logging import exception_log_forward

with exception_log_forward():
   <my glue code>
   ...
   <exception happens>
   <more of my code>
```
The `exception_log_forward` sets up a **context manager** that will trap exceptions and forward the exception/stack to CloudWatch for diagnosis. 

## Glue Job Local Testing
Glue Power without the Pain. Workbench manages the AWS Execution Role, so local API and Glue Jobs will have the same permissions/access. Also using the same Code as your notebooks or scripts makes creating and testing Glue Jobs a breeze.

```shell
export WORKBENCH_CONFIG=<your config>  # Only if not already set up
python my_glue_job.py --workbench-bucket <your bucket>
```

## Additional Resources
- Workbench Glue Jobs: [Workbench Glue](https://docs.google.com/presentation/d/1Bdcve27BDLbUkslZJAc1OrG6VkDopEtnL8Wh8HaLrco/edit?usp=sharing)
- Setting up Workbench on your AWS Account: [AWS Setup](../aws_setup/core_stack.md)
- Using Workbench for ML Pipelines: [Workbench API Classes](../api_classes/overview.md)

<img align="right" src="../images/scp.png" width="180">

- Workbench Core Classes: [Core Classes](../core_classes/overview.md)
- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
