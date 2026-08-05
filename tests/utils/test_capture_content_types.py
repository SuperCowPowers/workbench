"""Test that AWS accepts the data capture content types.

CreateEndpointConfig validates content types server-side, so a bad one only surfaces at
deploy time. This sends a real config carrying a nonexistent model: AWS validates the
request shape first, so reaching the "Could not find model" error means the content types
were accepted. The charset form is sent too, confirming AWS does reject what it should.
Every call fails, so no endpoint config is created.
"""

from botocore.exceptions import ClientError

from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.monitor_utils import CAPTURE_CSV_CONTENT_TYPES, CAPTURE_JSON_CONTENT_TYPES

MISSING_MODEL = "workbench-no-such-model"


def create_error(csv_content_types: list, json_content_types: list) -> str:
    """CreateEndpointConfig with the given content types, returning the error message."""
    sagemaker_client = AWSAccountClamp().boto3_session.client("sagemaker")
    try:
        sagemaker_client.create_endpoint_config(
            EndpointConfigName="workbench-content-type-probe",
            ProductionVariants=[
                {
                    "ModelName": MISSING_MODEL,
                    "VariantName": "probe",
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.c7i.large",
                }
            ],
            DataCaptureConfig={
                "EnableCapture": True,
                "InitialSamplingPercentage": 100,
                "DestinationS3Uri": "s3://workbench-content-type-probe/data_capture",
                "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
                "CaptureContentTypeHeader": {
                    "CsvContentTypes": csv_content_types,
                    "JsonContentTypes": json_content_types,
                },
            },
        )
    except ClientError as e:
        return e.response["Error"]["Message"]
    raise AssertionError("CreateEndpointConfig unexpectedly succeeded with a nonexistent model")


def test_capture_content_types_accepted():
    """AWS accepts our content types and rejects ones carrying parameters."""
    accepted = create_error(CAPTURE_CSV_CONTENT_TYPES, CAPTURE_JSON_CONTENT_TYPES)
    assert "captureContentTypeHeader" not in accepted, f"AWS rejected the content types: {accepted}"
    assert MISSING_MODEL in accepted

    rejected = create_error(["text/csv; charset=utf-8"], ["application/json; charset=utf-8"])
    assert "captureContentTypeHeader" in rejected
