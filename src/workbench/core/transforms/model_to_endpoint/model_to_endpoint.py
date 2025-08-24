"""ModelToEndpoint: Deploy an Endpoint for a Model"""

import time
from sagemaker import ModelPackage
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.model_monitor import DataCaptureConfig

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
        workbench_model = ModelCore(self.input_name)

        # Deploy the model
        self._deploy_model(workbench_model, **kwargs)

        # Add this endpoint to the set of registered endpoints for the model
        workbench_model.register_endpoint(self.output_name)

        # This ensures that the endpoint is ready for use
        time.sleep(5)  # We wait for AWS Lag
        end = EndpointCore(self.output_name)
        self.log.important(f"Endpoint {end.name} is ready for use")

    def _deploy_model(
        self,
        workbench_model: ModelCore,
        mem_size: int = 2048,
        max_concurrency: int = 5,
        data_capture: bool = False,
        capture_percentage: int = 100,
    ):
        """Internal Method: Deploy the Model

        Args:
            workbench_model(ModelCore): The Workbench ModelCore object to deploy
            mem_size(int): Memory size for serverless deployment
            max_concurrency(int): Max concurrency for serverless deployment
            data_capture(bool): Enable data capture during deployment
            capture_percentage(int): Percentage of data to capture. Defaults to 100.
        """
        # Grab the specified Model Package
        model_package_arn = workbench_model.model_package_arn()
        model_package = ModelPackage(
            role=self.workbench_role_arn,
            model_package_arn=model_package_arn,
            sagemaker_session=self.sm_session,
        )

        # Log the image that will be used for deployment
        inference_image = self.sm_client.describe_model_package(ModelPackageName=model_package_arn)[
            "InferenceSpecification"
        ]["Containers"][0]["Image"]
        self.log.important(f"Deploying Model Package: {self.input_name} with Inference Image: {inference_image}")

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()

        # Is this a serverless deployment?
        serverless_config = None
        if self.serverless:
            serverless_config = ServerlessInferenceConfig(
                memory_size_in_mb=mem_size,
                max_concurrency=max_concurrency,
            )

        # Configure data capture if requested (and not serverless)
        data_capture_config = None
        if data_capture and not self.serverless:
            # Set up the S3 path for data capture
            base_endpoint_path = f"{workbench_model.endpoints_s3_path}/{self.output_name}"
            data_capture_path = f"{base_endpoint_path}/data_capture"
            self.log.important(f"Configuring Data Capture --> {data_capture_path}")
            data_capture_config = DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=capture_percentage,
                destination_s3_uri=data_capture_path,
            )
        elif data_capture and self.serverless:
            self.log.warning(
                "Data capture is not supported for serverless endpoints. Skipping data capture configuration."
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
            data_capture_config=data_capture_config,
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
