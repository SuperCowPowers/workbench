import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.processing import Processor
from sagemaker.workflow.steps import ProcessingStep

# Get the Base Workbench Image URI
region = sagemaker.Session().boto_region_name
image_uri = "507740646243.dkr.ecr.us-west-2.amazonaws.com/sagworks_base:v0_4_29_amd64"

# Get the SageMaker Execution Role
account = "507740646243"
role = f"arn:aws:iam::{account}:role/Workbench-ExecutionRole"

# Define the environment variables
environment = {
    "WORKBENCH_BUCKET": "sandbox-workbench-artifacts",
}


# Define the processor
processor = Processor(
    role=role,
    image_uri=image_uri,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    env=environment,
)

# Define the various processing steps
datasource_step = ProcessingStep(
    name="DataSource",
    processor=processor,
    code="data_source.py",
)
featureset_step = ProcessingStep(
    name="FeatureSet",
    processor=processor,
    code="feature_set.py",
)
model_step = ProcessingStep(
    name="Model",
    processor=processor,
    code="model.py",
)
endpoint_step = ProcessingStep(
    name="Endpoint",
    processor=processor,
    code="endpoint.py",
)

# Define the pipeline
pipeline = Pipeline(
    name="HelloWorldPipeline",
    steps=[datasource_step, featureset_step, model_step, endpoint_step],
    sagemaker_session=sagemaker.Session(),
)

# Create the pipeline
pipeline.upsert(role_arn=role)

# Get the pipeline description
pipeline_description = pipeline.describe()

# Print the pipeline ARN
print(f"Pipeline ARN: {pipeline_description['PipelineArn']}")
