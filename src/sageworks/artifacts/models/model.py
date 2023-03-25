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

        # Grab our SageMaker Session
        self.sm_session = AWSSageWorksRoleManager().sagemaker_session()

        # All done
        self.log.info(f"Model Initialized: {model_name}")

    def check(self) -> bool:
        """Does the feature_set_name exist in the AWS Metadata?"""
        if self.latest_model is None:
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
        return self.model_tags

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
        """Delete the Model: Feature Group, Catalog Table, and S3 Storage Objects"""

        # Delete the Model and ensure that it gets deleted
        """
        remove_model = FeatureGroup(name=self.model_name, sagemaker_session=self.sm_session)
        remove_model.delete()
        self.ensure_model_deleted(remove_model)
        """

    def ensure_model_deleted(self, feature_group):
        """
        status = feature_group.describe().get("FeatureGroupStatus")
        while status == "Deleting":
            self.log.info("Model being Deleted...")
            time.sleep(5)
            try:
                status = feature_group.describe().get("FeatureGroupStatus")
            except Exception:  # FIXME
                break
        self.log.info(f"Model {feature_group.name} successfully deleted")
        """
        pass


# Simple test of the Model functionality
def test():
    """Test for Model Class"""

    # Grab a Model object and pull some information from it
    my_model = Model('solubility-regression')

    # Call the various methods

    # Lets do a check/validation of the Model
    assert(my_model.check())

    # Get the tags associated with this Model
    print(f"Tags: {my_model.tags()}")


if __name__ == "__main__":
    test()
