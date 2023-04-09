"""ModelToEndpoint: Deploy an Endpoint for a Model"""
import botocore
from sagemaker import ModelPackage
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer

# Local Imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.artifacts.models.model import Model


class ModelToEndpoint(Transform):
    """ModelToEndpoint: Deploy an Endpoint for a Model

    Common Usage:
        to_endpoint = ModelToEndpoint(model_uuid, endpoint_uuid)
        to_endpoint.set_output_tags(["aqsol", "public", "whatever"])
        to_endpoint.transform(delete_existing=True/False)
    """

    def __init__(self, model_uuid: str, endpoint_uuid: str):
        """ModelToEndpoint Initialization"""

        # Call superclass init
        super().__init__(model_uuid, endpoint_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.MODEL
        self.output_type = TransformOutput.ENDPOINT

    def transform_impl(self, delete_existing=True):
        """Compute a Feature Set based on RDKit Descriptors"""

        # Get the Model Package ARN for our input model
        model_package_arn = Model(self.input_uuid).model_arn()

        # Create a Model Package
        model_package = ModelPackage(role=self.sageworks_role_arn, model_package_arn=model_package_arn)

        # Check for delete
        if delete_existing:
            self.delete_endpoint()

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()

        # Deploy an Endpoint
        model_package.deploy(
            initial_instance_count=1,
            instance_type="ml.t2.medium",
            endpoint_name=self.output_uuid,
            serializer=CSVSerializer(),
            deserializer=CSVDeserializer(),
            tags=aws_tags,
        )

    def delete_endpoint(self):
        """Delete an existing Endpoint and it's Configuration"""
        # Delete endpoint (if it already exists)
        try:
            self.sm_client.delete_endpoint(EndpointName=self.output_uuid)
        except botocore.exceptions.ClientError:
            self.log.info(f"Endpoint {self.output_uuid} doesn't exist...")
        try:
            self.sm_client.delete_endpoint_config(EndpointConfigName=self.output_uuid)
        except botocore.exceptions.ClientError:
            self.log.info(f"Endpoint Config {self.output_uuid} doesn't exist...")


if __name__ == "__main__":
    """Exercise the ModelToEndpoint Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "aqsol-regression"
    output_uuid = "aqsol-regression-endpoint"
    to_endpoint = ModelToEndpoint(input_uuid, output_uuid)
    to_endpoint.set_output_tags(["aqsol", "public"])
    to_endpoint.transform()
