import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

# Get our Workbench Execution Role
role = AWSAccountClamp().aws_session.get_workbench_execution_role_arn()

# Hub Model configuration. https://huggingface.co/models
hub = {"HF_MODEL_ID": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "SM_NUM_GPUS": json.dumps(1)}

# Create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface", version="3.0.1"),
    env=hub,
    role=role,
)

# Deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    container_startup_health_check_timeout=300,
    endpoint_name="deepseek-8b",
)

# send request
predictor.predict(
    {
        "inputs": "Hi, what can you help me with?",
    }
)
