"""ModelToEndpoint: Deploy an Endpoint for a Model"""

import time
from sagemaker import ModelPackage
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker.serverless import ServerlessInferenceConfig

# Local Imports
from workbench.core.transforms.transform import Transform, TransformInput, TransformOutput
from workbench.core.artifacts.model_core import ModelCore
from workbench.core.artifacts.endpoint_core import EndpointCore
from workbench.core.artifacts.artifact import Artifact


class ModelToEndpoint(Transform):
    """ModelToEndpoint: Deploy an Endpoint for a Model

    Common Usage:
        ```python
        to_endpoint = ModelToEndpoint(model_uuid, endpoint_uuid)
        to_endpoint.set_output_tags(["aqsol", "public", "whatever"])
        to_endpoint.transform()
        ```
    """

    def __init__(self, model_uuid: str, endpoint_uuid: str, serverless: bool = True, instance: str = "ml.t2.medium"):
        """ModelToEndpoint Initialization
        Args:
            model_uuid(str): The UUID of the input Model
            endpoint_uuid(str): The UUID of the output Endpoint
            serverless(bool): Deploy the Endpoint in serverless mode (default: True)
            instance(str): The instance type to use for the Endpoint (default: "ml.t2.medium")
        """
        # Make sure the endpoint_uuid is a valid name
        Artifact.is_name_valid(endpoint_uuid, delimiter="-", lower_case=False)

        # Call superclass init
        super().__init__(model_uuid, endpoint_uuid)

        # Set up all my instance attributes
        self.serverless = serverless
        self.instance_type = "serverless" if serverless else instance
        self.input_type = TransformInput.MODEL
        self.output_type = TransformOutput.ENDPOINT

    def transform_impl(self):
        """Deploy an Endpoint for a Model"""

        # Delete endpoint (if it already exists)
        EndpointCore.managed_delete(self.output_uuid)

        # Get the Model Package ARN for our input model
        input_model = ModelCore(self.input_uuid)
        model_package_arn = input_model.model_package_arn()

        # Deploy the model
        self._deploy_model(model_package_arn)

        # Add this endpoint to the set of registered endpoints for the model
        input_model.register_endpoint(self.output_uuid)

        # This ensures that the endpoint is ready for use
        time.sleep(5)  # We wait for AWS Lag
        end = EndpointCore(self.output_uuid)
        self.log.important(f"Endpoint {end.uuid} is ready for use")

    def _deploy_model(self, model_package_arn: str):
        """Internal Method: Deploy the Model

        Args:
            model_package_arn(str): The Model Package ARN used to deploy the Endpoint
        """
        # Grab the specified Model Package
        model_package = ModelPackage(
            role=self.workbench_role_arn,
            model_package_arn=model_package_arn,
            sagemaker_session=self.sm_session,
        )

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()

        # Is this a serverless deployment?
        serverless_config = None
        if self.serverless:
            serverless_config = ServerlessInferenceConfig(
                memory_size_in_mb=2048,
                max_concurrency=5,
            )

        # Deploy the Endpoint
        self.log.important(f"Deploying the Endpoint {self.output_uuid}...")
        model_package.deploy(
            initial_instance_count=1,
            instance_type=self.instance_type,
            serverless_inference_config=serverless_config,
            endpoint_name=self.output_uuid,
            serializer=CSVSerializer(),
            deserializer=CSVDeserializer(),
            tags=aws_tags,
        )

    def post_transform(self, **kwargs):
        """Post-Transform: Calling onboard() for the Endpoint"""
        self.log.info("Post-Transform: Calling onboard() for the Endpoint...")

        # Onboard the Endpoint
        output_endpoint = EndpointCore(self.output_uuid)
        output_endpoint.onboard_with_args(input_model=self.input_uuid)


if __name__ == "__main__":
    """Exercise the ModelToEndpoint Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "abalone-regression"
    output_uuid = f"{input_uuid}"
    to_endpoint = ModelToEndpoint(input_uuid, output_uuid, serverless=True)
    to_endpoint.set_output_tags(["abalone", "public"])
    to_endpoint.transform()
