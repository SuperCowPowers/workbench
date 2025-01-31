import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

# Get our Workbench Execution Role
role = AWSAccountClamp().aws_session.get_workbench_execution_role_arn()

# Hub Model configuration. https://huggingface.co/models
hub = {
    "HF_MODEL_ID": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "HF_NUM_CORES": "2",
    "HF_AUTO_CAST_TYPE": "bf16",
    "MAX_BATCH_SIZE": "8",
    "MAX_INPUT_TOKENS": "3686",
    "MAX_TOTAL_TOKENS": "4096",
}


region = boto3.Session().region_name
image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.2-optimum0.0.27-neuronx-py310-ubuntu22.04"

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=image_uri,
    env=hub,
    role=role,
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    endpoint_name="deepseek-8b",
    initial_instance_count=1,
    instance_type="ml.inf2.xlarge",
    container_startup_health_check_timeout=1800,
    volume_size=512,
)

# send request
predictor.predict(
    {
        "inputs": "What is is the capital of France?",
        "parameters": {
            "do_sample": True,
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        },
    }
)
