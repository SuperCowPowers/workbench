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
        to_endpoint = ModelToEndpoint(model_name, endpoint_name)
        to_endpoint.set_output_tags(["aqsol", "public", "whatever"])
        to_endpoint.transform()
        ```
    """

    def __init__(self, model_name: str, endpoint_name: str, serverless: bool = True, instance: str = "ml.t2.medium"):
        """ModelToEndpoint Initialization
        Args:
            model_name(str): The Name of the input Model
            endpoint_name(str): The Name of the output Endpoint
            serverless(bool): Deploy the Endpoint in serverless mode (default: True)
            instance(str): The instance type to use for the Endpoint (default: "ml.t2.medium")
        """
        # Make sure the endpoint_name is a valid name
        Artifact.is_name_valid(endpoint_name, delimiter="-", lower_case=False)

        # Call superclass init
        super().__init__(model_name, endpoint_name)

        # Set up all my instance attributes
        self.serverless = serverless
        self.instance_type = "serverless" if serverless else instance
        self.input_type = TransformInput.MODEL
        self.output_type = TransformOutput.ENDPOINT

    def transform_impl(self, **kwargs):
        """Deploy an Endpoint for a Model"""

        # Delete endpoint (if it already exists)
        EndpointCore.managed_delete(self.output_name)

        # Get the Model Package ARN for our input model
        input_model = ModelCore(self.input_name)
        model_package_arn = input_model.model_package_arn()

        # Deploy the model
        self._deploy_model(model_package_arn, **kwargs)

        # Add this endpoint to the set of registered endpoints for the model
        input_model.register_endpoint(self.output_name)

        # This ensures that the endpoint is ready for use
        time.sleep(5)  # We wait for AWS Lag
        end = EndpointCore(self.output_name)
        self.log.important(f"Endpoint {end.name} is ready for use")

    def _deploy_model(self, model_package_arn: str, mem_size: int = 2048, max_concurrency: int = 5):
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
                memory_size_in_mb=mem_size,
                max_concurrency=max_concurrency,
            )

        # Deploy the Endpoint
        self.log.important(f"Deploying the Endpoint {self.output_name}...")
        model_package.deploy(
            initial_instance_count=1,
            instance_type=self.instance_type,
            serverless_inference_config=serverless_config,
            endpoint_name=self.output_name,
            serializer=CSVSerializer(),
            deserializer=CSVDeserializer(),
            tags=aws_tags,
        )

    def post_transform(self, **kwargs):
        """Post-Transform: Calling onboard() for the Endpoint"""
        self.log.info("Post-Transform: Calling onboard() for the Endpoint...")

        # Onboard the Endpoint
        output_endpoint = EndpointCore(self.output_name)
        output_endpoint.onboard_with_args(input_model=self.input_name)


if __name__ == "__main__":
    """Exercise the ModelToEndpoint Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_name = "abalone-regression"
    output_name = f"{input_name}"
    to_endpoint = ModelToEndpoint(input_name, output_name, serverless=True)
    to_endpoint.set_output_tags(["abalone", "public"])
    to_endpoint.transform()
