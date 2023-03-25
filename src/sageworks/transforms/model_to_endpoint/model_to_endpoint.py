"""ModelToEndpoint: Deploy an Endpoint for a Model"""
import botocore
from sagemaker import ModelPackage
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer

# Local Imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.artifacts.models.model import Model


class ModelToEndpoint(Transform):
    def __init__(self, input_uuid: str = None, output_uuid: str = None):
        """ModelToEndpoint: Deploy an Endpoint for a Model"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.MODEL
        self.output_type = TransformOutput.ENDPOINT

    def transform_impl(self, delete_existing=True):
        """Compute a Feature Set based on RDKit Descriptors"""

        # Get the Model Package ARN for our input model
        model_package_arn = Model(self.input_uuid).model_package_arn

        # Create a Model Package
        model_package = ModelPackage(role=self.sageworks_role_arn, model_package_arn=model_package_arn)

        # Check for delete
        if delete_existing:
            self.delete_endpoint()

        # Deploy an Endpoint
        model_package.deploy(initial_instance_count=1, instance_type='ml.t2.medium',
                             endpoint_name=self.output_uuid,
                             serializer=CSVSerializer(),
                             deserializer=CSVDeserializer())

    def delete_endpoint(self):
        """Delete an existing Endpoint and it's Configuration"""
        # Delete endpoint (if it already exists)
        sm_client = self.sm_session.boto_session.client("sagemaker")
        try:
            sm_client.delete_endpoint(EndpointName=self.output_uuid)
        except botocore.exceptions.ClientError:
            self.log.info(f"Endpoint {self.output_uuid} doesn't exist...")
        try:
            sm_client.delete_endpoint_config(EndpointConfigName=self.output_uuid)
        except botocore.exceptions.ClientError:
            self.log.info(f"Endpoint Config {self.output_uuid} doesn't exist...")


# Simple test of the ModelToEndpoint functionality
def test():
    """Test the ModelToEndpoint Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = 'solubility-regression'
    output_uuid = 'solubility-regression-endpoint'
    ModelToEndpoint(input_uuid, output_uuid).transform()


if __name__ == "__main__":
    test()
