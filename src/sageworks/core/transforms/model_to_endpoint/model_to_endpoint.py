"""ModelToEndpoint: Deploy an Endpoint for a Model"""

import time
from datetime import datetime
from sagemaker import ModelPackage
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from botocore.exceptions import ClientError

# Local Imports
from sageworks.core.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.core.artifacts.model_core import ModelCore
from sageworks.core.artifacts.endpoint_core import EndpointCore
from sageworks.core.artifacts.artifact import Artifact


class ModelToEndpoint(Transform):
    """ModelToEndpoint: Deploy an Endpoint for a Model

    Common Usage:
        ```
        to_endpoint = ModelToEndpoint(model_uuid, endpoint_uuid)
        to_endpoint.set_output_tags(["aqsol", "public", "whatever"])
        to_endpoint.transform()
        ```
    """

    def __init__(self, model_uuid: str, endpoint_uuid: str, serverless: bool = True):
        """ModelToEndpoint Initialization
        Args:
            model_uuid(str): The UUID of the input Model
            endpoint_uuid(str): The UUID of the output Endpoint
            serverless(bool): Deploy the Endpoint in serverless mode (default: True)
        """

        # Make sure the endpoint_uuid is a valid name
        Artifact.ensure_valid_name(endpoint_uuid, delimiter="-")

        # Call superclass init
        super().__init__(model_uuid, endpoint_uuid)

        # Set up all my instance attributes
        self.instance_type = "serverless" if serverless else "ml.t2.medium"
        self.input_type = TransformInput.MODEL
        self.output_type = TransformOutput.ENDPOINT

    def transform_impl(self):
        """Deploy an Endpoint for a Model"""

        # Delete endpoint (if it already exists)
        existing_endpoint = EndpointCore(self.output_uuid, force_refresh=True)
        if existing_endpoint.exists():
            existing_endpoint.delete()

        # Get the Model Package ARN for our input model
        input_model = ModelCore(self.input_uuid)
        model_package_arn = input_model.model_package_arn()

        # Will this be a Serverless Endpoint?
        if self.instance_type == "serverless":
            self._serverless_deploy(model_package_arn)
        else:
            self._realtime_deploy(model_package_arn)

        # Add this endpoint to the set of registered endpoints for the model
        input_model.register_endpoint(self.output_uuid)

        # This ensures that the endpoint is ready for use
        time.sleep(5)  # We wait for AWS Lag
        end = EndpointCore(self.output_uuid, force_refresh=True)
        self.log.important(f"Endpoint {end.uuid} is ready for use")

    def _realtime_deploy(self, model_package_arn: str):
        """Internal Method: Deploy the Realtime Endpoint

        Args:
            model_package_arn(str): The Model Package ARN used to deploy the Endpoint
        """
        # Create a Model Package
        model_package = ModelPackage(role=self.sageworks_role_arn, model_package_arn=model_package_arn)

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()

        # Deploy a Realtime Endpoint
        model_package.deploy(
            initial_instance_count=1,
            instance_type=self.instance_type,
            endpoint_name=self.output_uuid,
            serializer=CSVSerializer(),
            deserializer=CSVDeserializer(),
            tags=aws_tags,
        )

    def _serverless_deploy(self, model_package_arn, mem_size=2048, max_concurrency=5, wait=True):
        """Internal Method: Deploy a Serverless Endpoint

        Args:
            mem_size(int): Memory size in MB (default: 2048)
            max_concurrency(int): Max concurrency (default: 5)
            wait(bool): Wait for the Endpoint to be ready (default: True)
        """
        model_name = self.input_uuid
        endpoint_name = self.output_uuid
        aws_tags = self.get_aws_tags()

        # Create Low Level Model Resource (Endpoint Config below references this Model Resource)
        # Note: Since model is internal to the endpoint we'll add a timestamp (just like SageMaker does)
        datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
        model_name = f"{model_name}-{datetime_str}"
        self.log.info(f"Creating Low Level Model: {model_name}...")
        self.sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "ModelPackageName": model_package_arn,
            },
            ExecutionRoleArn=self.sageworks_role_arn,
            Tags=aws_tags,
        )

        # Create Endpoint Config
        self.log.info(f"Creating Endpoint Config {endpoint_name}...")
        try:
            self.sm_client.create_endpoint_config(
                EndpointConfigName=endpoint_name,
                ProductionVariants=[
                    {
                        "ServerlessConfig": {"MemorySizeInMB": mem_size, "MaxConcurrency": max_concurrency},
                        "ModelName": model_name,
                        "VariantName": "AllTraffic",
                    }
                ],
            )
        except ClientError as e:
            # Already Exists: Check if ValidationException and existing endpoint configuration
            if (
                e.response["Error"]["Code"] == "ValidationException"
                and "already existing endpoint configuration" in e.response["Error"]["Message"]
            ):
                self.log.warning("Endpoint configuration already exists: Deleting and retrying...")
                self.sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
                self.sm_client.create_endpoint_config(
                    EndpointConfigName=endpoint_name,
                    ProductionVariants=[
                        {
                            "ServerlessConfig": {"MemorySizeInMB": mem_size, "MaxConcurrency": max_concurrency},
                            "ModelName": model_name,
                            "VariantName": "AllTraffic",
                        }
                    ],
                )

        # Create Endpoint
        self.log.info(f"Creating Serverless Endpoint {endpoint_name}...")
        self.sm_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_name, Tags=self.get_aws_tags()
        )

        # Wait for Endpoint to be ready
        if not wait:
            self.log.important(f"Endpoint {endpoint_name} is being created...")
        else:
            self.log.important(f"Waiting for Endpoint {endpoint_name} to be ready...")
            describe_endpoint_response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            while describe_endpoint_response["EndpointStatus"] == "Creating":
                time.sleep(30)
                describe_endpoint_response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
                self.log.info(describe_endpoint_response["EndpointStatus"])
            status = describe_endpoint_response["EndpointStatus"]
            if status != "InService":
                msg = f"Endpoint {endpoint_name} failed to be created. Status: {status}"
                details = describe_endpoint_response["FailureReason"]
                self.log.critical(msg)
                self.log.critical(details)
                raise Exception(msg)
            self.log.important(f"Endpoint {endpoint_name} is now {status}...")

    def post_transform(self, **kwargs):
        """Post-Transform: Calling onboard() for the Endpoint"""
        self.log.info("Post-Transform: Calling onboard() for the Endpoint...")

        # Onboard the Endpoint
        output_endpoint = EndpointCore(self.output_uuid, force_refresh=True)
        output_endpoint.onboard()


if __name__ == "__main__":
    """Exercise the ModelToEndpoint Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "abalone-regression"
    output_uuid = "abalone-regression-end"
    to_endpoint = ModelToEndpoint(input_uuid, output_uuid)
    to_endpoint.set_output_tags(["aqsol", "public"])
    to_endpoint.transform()
