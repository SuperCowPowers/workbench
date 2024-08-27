import sagemaker
from sagemaker.session import Session as SageSession
from sagemaker import image_uris


def get_image_uri_with_digest(framework, region, version, sm_session: SageSession):
    # Retrieve the base image URI using sagemaker SDK
    base_image_uri = image_uris.retrieve(
        framework=framework, region=region, version=version, sagemaker_session=sm_session
    )
    print(f"Base Image URI: {base_image_uri}")

    # Extract repository name and image tag from the base image URI
    repo_uri, image_tag = base_image_uri.split(":")
    repository_name = repo_uri.split("/")[-1]

    # Use AWS CLI to get image details and find the digest
    ecr_client = sm_session.boto_session.client("ecr", region_name=region)
    response = ecr_client.describe_images(
        repositoryName=repository_name,
        imageIds=[
            {"imageTag": image_tag},
        ],
    )
    if "imageDetails" in response and len(response["imageDetails"]) > 0:
        image_digest = response["imageDetails"][0]["imageDigest"]
        full_image_uri = f"{repo_uri}@{image_digest}"
        return full_image_uri
    else:
        raise ValueError("Image details not found for the specified tag.")


if __name__ == "__main__":
    from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp

    # Get the image URI with digest
    framework = "sklearn"
    region = "us-west-2"
    version = "1.2-1"

    # Grab a sagemaker session from SageWorks
    sm_session = AWSAccountClamp().sagemaker_session()

    try:
        full_image_uri = get_image_uri_with_digest(framework, region, version, sm_session)
        print(f"Full Image URI with Digest: {full_image_uri}")
    except Exception as e:
        print(f"Error: {e}")
