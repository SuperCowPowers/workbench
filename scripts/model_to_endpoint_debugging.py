import sagemaker
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded values
MODEL_PACKAGE_GROUP_NAME = "abalone-regression"
ENDPOINT_NAME = f"{MODEL_PACKAGE_GROUP_NAME}"
MEM_SIZE = 4096
MAX_CONCURRENCY = 10
ROLE_ARN = "arn:aws:iam::507740646243:role/Workbench-ExecutionRole"

# Get a standard SageMaker session
session = sagemaker.Session()


# Get the latest approved model package
def get_latest_model_package(model_package_group_name):
    model_packages = session.sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
    )
    if not model_packages["ModelPackageSummaryList"]:
        raise ValueError(f"No approved model packages found in group: {model_package_group_name}")

    latest_model_package_arn = model_packages["ModelPackageSummaryList"][0]["ModelPackageArn"]

    # Describe the model package to get details
    model_package_details = session.sagemaker_client.describe_model_package(ModelPackageName=latest_model_package_arn)

    logger.info(f"Model Package Details: {model_package_details}")
    return latest_model_package_arn


# Delete existing model if it exists
def delete_model_if_exists(model_name):
    try:
        session.sagemaker_client.delete_model(ModelName=model_name)
        logger.info(f"Deleted existing model: {model_name}")
    except ClientError as e:
        if (
            e.response["Error"]["Code"] == "ValidationException"
            and "Could not find model" in e.response["Error"]["Message"]
        ):
            logger.info(f"Model {model_name} does not exist, no need to delete")
        else:
            raise


# Create a serverless endpoint
def create_serverless_endpoint(model_package_arn, endpoint_name, role_arn, mem_size, max_concurrency):
    try:
        # Define the model name
        model_name = f"{endpoint_name}-model"

        # Delete the existing model if it exists
        delete_model_if_exists(model_name)

        # Create the model from the model package
        logger.info(f"Creating model with ARN: {model_package_arn}")
        session.sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "ModelPackageName": model_package_arn,
            },
            ExecutionRoleArn=role_arn,
        )

        # Define the endpoint configuration name
        endpoint_config_name = f"{endpoint_name}-config"

        # Create the endpoint configuration
        logger.info(
            f"Creating endpoint configuration with memory size: {mem_size} and max concurrency: {max_concurrency}"
        )
        session.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "ServerlessConfig": {"MemorySizeInMB": mem_size, "MaxConcurrency": max_concurrency},
                    "ModelName": model_name,
                    "VariantName": "AllTraffic",
                }
            ],
        )

        # Create the endpoint
        logger.info(f"Creating endpoint: {endpoint_name}")
        response = session.create_endpoint(
            endpoint_name=endpoint_name, config_name=endpoint_config_name, wait=True, live_logging=True
        )

        logger.info(f"Endpoint creation response: {response}")
        logger.info(f"Endpoint {endpoint_name} created successfully.")

        # Check the endpoint logs
        endpoint_logs = session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Endpoint Logs: {endpoint_logs}")

    except ClientError as e:
        logger.error(f"ClientError: {e.response['Error']['Code']}, {e.response['Error']['Message']}")
        if "LogResult" in e.response:
            log_result = e.response["LogResult"]
            logger.error(f"LogResult: {log_result}")
        raise
    except Exception as e:
        logger.error(f"Failed to create serverless endpoint: {str(e)}")
        raise


# Main execution
if __name__ == "__main__":
    try:
        latest_model_package_arn = get_latest_model_package(MODEL_PACKAGE_GROUP_NAME)
        logger.info(f"Latest model package ARN: {latest_model_package_arn}")
        create_serverless_endpoint(latest_model_package_arn, ENDPOINT_NAME, ROLE_ARN, MEM_SIZE, MAX_CONCURRENCY)
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise
