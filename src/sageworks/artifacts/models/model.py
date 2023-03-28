"""Model: SageWorks Model Class"""
import json
from datetime import datetime


# SageWorks Imports
from sageworks.artifacts.artifact import Artifact
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory, AWSServiceBroker
from sageworks.aws_service_broker.aws_sageworks_role_manager import AWSSageWorksRoleManager


class Model(Artifact):

    def __init__(self, model_name):
        """Model: SageWorks Model Class

        Args:
            model_name (str): Name of Model in SageWorks.
        """
        # Call SuperClass Initialization
        super().__init__()

        # Grab an AWS Metadata Broker object and pull information for Models
        self.model_name = model_name
        self.aws_meta = AWSServiceBroker()
        self.model_meta = self.aws_meta.get_metadata(ServiceCategory.MODELS).get(self.model_name)
        if self.model_meta is None:
            # Base Class Initialization
            Artifact.__init__(self)
            self.log.warning(f"Could not find model {self.model_name} within current visibility scope")
        else:
            self.latest_model = self.model_meta[0]
            self.model_package_arn = self.latest_model['ModelPackageDetails']['ModelPackageArn']

            # Pull Model Package Description Data
            try:
                self.model_info = json.loads(self.latest_model['ModelPackageDescription'])
                self.input = self.model_info['input']
                self.description = self.model_info['info']
                self.model_tags = self.model_info['tags']
            except (json.JSONDecoder, KeyError):
                self.model_info = self.latest_model['ModelPackageDescription']
                self.input = '-'
                self.description = self.model_info
                self.model_tags = '-'

        # All done
        self.log.info(f"Model Initialized: {model_name}")

    def check(self) -> bool:
        """Does the feature_set_name exist in the AWS Metadata?"""
        if self.model_meta is None:
            self.log.critical(f'Model.check() {self.model_name} not found in AWS Metadata!')
            return False
        return True

    def uuid(self) -> str:
        """The SageWorks Unique Identifier"""
        return self.model_name

    def size(self) -> bool:
        """Return the size of this data in MegaBytes"""
        return 0

    def meta(self):
        """Get the metadata for this artifact"""
        return self.latest_model

    def add_tag(self):
        """Get the tags for this artifact"""
        return []

    def tags(self):
        """Get the tags for this artifact"""
        return getattr(self, 'model_tags', [])

    def aws_url(self):
        """The AWS URL for looking at/querying this data source"""
        return 'https://us-west-2.console.aws.amazon.com/athena/home'

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.latest_model['CreationTime']

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        return self.latest_model['CreationTime']

    def delete(self):
        """Delete the Model Packages and the Model Group"""

        # If we don't have meta then the model probably doesn't exist
        if self.model_meta is None:
            self.log.info(f"Model {self.model_name} doesn't appear to exist...")
            return

        # First delete the Model Packages within the Model Group
        for model in self.model_meta:
            self.log.info(f"Deleting Model Package {model['ModelPackageArn']}...")
            self.sm_session.sagemaker_client.delete_model_package(ModelPackageName=model['ModelPackageArn'])

        # Now delete the Model Package Group
        self.log.info(f"Deleting Model Group {self.model_name}...")
        self.sm_session.sagemaker_client.delete_model_package_group(ModelPackageGroupName=self.model_name)


# Simple test of the Model functionality
def test():
    """Test for Model Class"""

    # Grab a Model object and pull some information from it
    my_model = Model('abalone-regression')

    # Call the various methods

    # Lets do a check/validation of the Model
    print(f"Model Check: {my_model.check()}")

    # Get the tags associated with this Model
    print(f"Tags: {my_model.tags()}")

    # Delete the Model
    my_model.delete()


if __name__ == "__main__":
    test()
