import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.processing import Processor
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import SKLearnProcessor

# Get the Base SageWorks Image URI
region = sagemaker.Session().boto_region_name
image_uri = "507740646243.dkr.ecr.us-west-2.amazonaws.com/sagworks_base:v0_4_29_amd64"

# Get the SageMaker Execution Role
account = "507740646243"
role = f"arn:aws:iam::{account}:role/SageWorks-ExecutionRole"

environment = {
    "SAGEWORKS_BUCKET": "sandbox-sageworks-artifacts",
}

# Define the processor
processor = Processor(
    role=role,
    image_uri=image_uri,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    env=environment,
)

# Define processing step for data preprocessing
"""
processor = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="sklearn-processing",
    role=role,
)
"""

# Define the processing step
processing_step = ProcessingStep(
    name="AllStepsInOne",
    processor=processor,
    code="hello.py",
)

# Define the pipeline
pipeline = Pipeline(
    name="SageWorksMLPipeline",
    steps=[processing_step],
    sagemaker_session=sagemaker.Session(),
)

# Create the pipeline
pipeline.upsert(role_arn=role)

# Get the pipeline description
pipeline_description = pipeline.describe()

# Print the pipeline ARN
print(f"Pipeline ARN: {pipeline_description['PipelineArn']}")
