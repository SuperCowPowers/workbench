"""Lightweight boto3 session helper with optional Workbench role assumption.

This is the endpoint-safe / service-safe counterpart to
:class:`workbench.core.cloud_platform.aws.aws_session.AWSSession` (which
requires a Workbench config file and uses refreshable credentials).

* When ``running_as_service()`` is ``True`` (SageMaker inference container,
  ECS, Lambda, Glue, Docker), returns a plain ``boto3.Session()`` so the
  container's own attached IAM role provides credentials. No Workbench
  config or role-assumption is attempted.
* When running locally, attempts to assume ``Workbench-ExecutionRole``. On
  failure, falls back to the default session.

Used by ``ParameterStoreCore``, ``InferenceStore``, and the raw
``DFStoreCore`` so those classes work in both endpoint and orchestration
contexts without requiring a Workbench config.
"""

import boto3
import logging
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

from workbench.utils.execution_environment import running_as_service

log = logging.getLogger("workbench")


def get_boto3_session() -> boto3.Session:
    """Get a boto3 session, optionally assuming the Workbench execution role.

    Returns:
        boto3.Session: A boto3 session (with assumed role credentials when running locally).
    """
    session = boto3.Session()

    # Only assume Workbench role when running locally (not as a service)
    if not running_as_service():
        role = "Workbench-ExecutionRole"
        try:
            account_id = session.client("sts").get_caller_identity()["Account"]
            assumed_role = session.client("sts").assume_role(
                RoleArn=f"arn:aws:iam::{account_id}:role/{role}", RoleSessionName="WorkbenchSession"
            )
            credentials = assumed_role["Credentials"]
            session = boto3.Session(
                aws_access_key_id=credentials["AccessKeyId"],
                aws_secret_access_key=credentials["SecretAccessKey"],
                aws_session_token=credentials["SessionToken"],
            )
        except (ClientError, NoCredentialsError, PartialCredentialsError) as e:
            # Log the failure and proceed with the default session
            log.important(f"Failed to assume Workbench role: {e}. Using default session.")
    return session


if __name__ == "__main__":
    """Exercise the boto session helper"""
    from workbench.core.parameter_store_core import ParameterStoreCore

    boto3_session = get_boto3_session()

    print("\nSageMaker Models:")
    sagemaker_client = boto3_session.client("sagemaker")
    response = sagemaker_client.list_models()
    for model in response["Models"]:
        print(model["ModelName"])

    param_key = "/workbench/config/workbench_bucket"
    workbench_bucket = ParameterStoreCore().get(param_key)
    if workbench_bucket is None:
        raise ValueError(f"Set '{param_key}' in Parameter Store.")
    s3_client = boto3_session.client("s3")
    try:
        response = s3_client.list_objects_v2(Bucket=workbench_bucket, MaxKeys=10)
        if "Contents" in response:
            print(f"\nFirst 10 objects in '{workbench_bucket}':")
            for obj in response["Contents"]:
                print(f"  {obj['Key']} ({obj['Size']} bytes)")
        else:
            print(f"\nBucket '{workbench_bucket}' is empty or no objects found.")
    except ClientError as e:
        print(f"Failed to access workbench bucket '{workbench_bucket}': {e}")
