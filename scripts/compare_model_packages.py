import boto3
from pprint import pprint


def get_model_package_arn(model_package_name):
    sagemaker_client = boto3.client("sagemaker")
    response = sagemaker_client.list_model_packages(ModelPackageGroupName=model_package_name)
    if not response["ModelPackageSummaryList"]:
        raise ValueError(f"Model package {model_package_name} does not exist.")
    return response["ModelPackageSummaryList"][0]["ModelPackageArn"]


def get_model_package_details(model_package_arn):
    sagemaker_client = boto3.client("sagemaker")
    response = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
    return response


def compare_model_packages(model_package_name_1, model_package_name_2):
    arn_1 = get_model_package_arn(model_package_name_1)
    arn_2 = get_model_package_arn(model_package_name_2)

    details_1 = get_model_package_details(arn_1)
    details_2 = get_model_package_details(arn_2)

    # Compare container image details
    container_image_1 = details_1["InferenceSpecification"]["Containers"][0]["Image"]
    container_image_2 = details_2["InferenceSpecification"]["Containers"][0]["Image"]

    print(f"Container Image Comparison:\nModel A: {container_image_1}\nModel B: {container_image_2}")

    # Compare model artifact locations
    model_artifact_1 = details_1["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    model_artifact_2 = details_2["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    print(f"Model Artifacts Comparison:\nModel A: {model_artifact_1}\nModel B: {model_artifact_2}")

    # Compare SAGEMAKER_SUBMIT_DIRECTORY
    submit_dir_1 = (
        details_1.get("InferenceSpecification", {})
        .get("Containers", [{}])[0]
        .get("Environment", {})
        .get("SAGEMAKER_SUBMIT_DIRECTORY", "Not Defined")
    )
    submit_dir_2 = (
        details_2.get("InferenceSpecification", {})
        .get("Containers", [{}])[0]
        .get("Environment", {})
        .get("SAGEMAKER_SUBMIT_DIRECTORY", "Not Defined")
    )

    print(f"SAGEMAKER_SUBMIT_DIRECTORY Comparison:\nModel A: {submit_dir_1}\nModel B: {submit_dir_2}")

    # Compare endpoint configurations (for serverless)
    serverless_config_1 = details_1.get("ServerlessConfig", None)
    serverless_config_2 = details_2.get("ServerlessConfig", None)

    print("Serverless Config Comparison:")
    if serverless_config_1:
        print(f"Model A: {serverless_config_1}")
    else:
        print("Model A: Not Defined")

    if serverless_config_2:
        print(f"Model B: {serverless_config_2}")
    else:
        print("Model B: Not Defined")

    # Dump the full model package details
    print("Model Package Details Comparison:")
    print("Model A:")
    pprint(details_1)
    print("\n\nModel B:")
    pprint(details_2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python compare_model_packages.py <ModelPackageName1> <ModelPackageName2>")
        sys.exit(1)

    model_package_name_1 = sys.argv[1]
    model_package_name_2 = sys.argv[2]

    compare_model_packages(model_package_name_1, model_package_name_2)
