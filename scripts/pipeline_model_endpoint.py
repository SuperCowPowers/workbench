from sagemaker.pipeline import PipelineModel

# Workbench imports
from workbench.api import Model
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

session = AWSAccountClamp().sagemaker_session()
role = AWSAccountClamp().aws_session.get_workbench_execution_role_arn()

# Our three models
taut_model = Model("tautomerize-v0").latest_model_object()
md_model = Model("smiles-to-md-v0").latest_model_object()
sol_model = Model("aqsol-mol-class").latest_model_object()

# Create a pipeline model that chains the three models
# pipeline_model = PipelineModel(name="pipeline-model", models=[taut_model, md_model, sol_model], role=role, sagemaker_session=session)
pipeline_model = PipelineModel(name="pipeline-model", models=[taut_model, md_model], role=role, sagemaker_session=session)

# Deploy the pipeline endpoint
predictor = pipeline_model.deploy(initial_instance_count=1, instance_type="ml.m5.large")
