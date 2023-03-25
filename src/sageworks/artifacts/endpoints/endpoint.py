"""Endpoint: SageWorks Endpoint Class"""
from datetime import datetime
import botocore


# SageWorks Imports
from sageworks.artifacts.artifact import Artifact
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory, AWSServiceBroker
from sageworks.aws_service_broker.aws_sageworks_role_manager import AWSSageWorksRoleManager


class Endpoint(Artifact):

    def __init__(self, endpoint_name):
        """Endpoint: SageWorks Endpoint Class

        Args:
            endpoint_name (str): Name of Endpoint in SageWorks
        """
        # Call SuperClass Initialization
        super().__init__()

        # Grab an AWS Metadata Broker object and pull information for Endpoints
        self.endpoint_name = endpoint_name
        self.aws_meta = AWSServiceBroker()
        self.endpoint_meta = self.aws_meta.get_metadata(ServiceCategory.ENDPOINTS).get(self.endpoint_name)

        # Grab our SageMaker Session
        self.sm_session = AWSSageWorksRoleManager().sagemaker_session()

        # All done
        self.log.info(f"Endpoint Initialized: {endpoint_name}")

    def check(self) -> bool:
        """Does the feature_set_name exist in the AWS Metadata?"""
        if self.endpoint_meta is None:
            self.log.critical(f'Endpoint.check() {self.endpoint_name} not found in AWS Metadata!')
            return False
        return True

    def uuid(self) -> str:
        """The SageWorks Unique Identifier"""
        return self.endpoint_name

    def size(self) -> bool:
        """Return the size of this data in MegaBytes"""
        return 0

    def meta(self):
        """Get the metadata for this artifact"""
        return self.endpoint_meta

    def add_tag(self):
        """Get the tags for this artifact"""
        return []

    def tags(self):
        """Get the tags for this artifact"""
        return []

    def aws_url(self):
        """The AWS URL for looking at/querying this data source"""
        return 'https://us-west-2.console.aws.amazon.com/athena/home'

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.endpoint_meta['CreationTime']

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        return self.endpoint_meta['LastModifiedTime']

    def delete(self):
        """Delete the Endpoint and Endpoint Config"""

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


# Simple test of the Endpoint functionality
def test():
    """Test for Endpoint Class"""

    # Grab a Endpoint object and pull some information from it
    my_endpoint = Endpoint('solubility-regression-endpoint')

    # Call the various methods

    # Let's do a check/validation of the Endpoint
    assert(my_endpoint.check())

    # Creation/Modification Times
    print(my_endpoint.created())
    print(my_endpoint.modified())

    # Get the tags associated with this Endpoint
    print(f"Tags: {my_endpoint.tags()}")


if __name__ == "__main__":
    test()
