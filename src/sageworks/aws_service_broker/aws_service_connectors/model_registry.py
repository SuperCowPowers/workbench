"""ModelRegistry: Helper Class for the AWS Model Registry Service"""

import sys
import argparse

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.connector import Connector
from sageworks.utils.aws_utils import list_tags_with_throttle, compute_size


class ModelRegistry(Connector):
    def __init__(self):
        """ModelRegistry: Helper Class for the AWS Model Registry Service"""
        # Call SuperClass Initialization
        super().__init__()

        # Set up our internal data storage
        self.model_data = {}
        self.model_package_group_arns = {}

    def check(self) -> bool:
        """Check if we can reach/connect to this AWS Service"""
        try:
            self.sm_client.list_model_package_groups()
            return True
        except Exception as e:
            self.log.critical(f"Error connecting to AWS Model Registry Service: {e}")
            return False

    def refresh(self):
        """Refresh all the Model Registry Data from SageMaker"""
        self.log.info("Refreshing Model Registry Data from SageMaker...")
        _model_groups = self.sm_client.list_model_package_groups(MaxResults=100)["ModelPackageGroupSummaryList"]
        _mg_names = [model_group["ModelPackageGroupName"] for model_group in _model_groups]

        # Grab the ModelPackageGroupARNs (we'll use them later to store in the model_data)
        self.model_package_group_arns = {g["ModelPackageGroupName"]: g["ModelPackageGroupArn"] for g in _model_groups}

        # Get the details for each Model Group and convert to a data structure with direct lookup
        self.model_data = {name: self._model_group_details(name) for name in _mg_names}

        # Additional details under the sageworks_meta section for each Model Group
        for mg_name in _mg_names:
            sageworks_meta = list_tags_with_throttle(self.model_package_group_arns[mg_name], self.sm_session)
            # Model groups have a list of models
            for model_info in self.model_data[mg_name]:
                model_info["sageworks_meta"] = sageworks_meta

        # Track the size of the metadata
        for key in self.model_data.keys():
            self.metadata_size_info[key] = compute_size(self.model_data[key])

    def summary(self) -> dict:
        """Return a summary of all the AWS Model Registry Groups"""
        return self.model_data

    def model_group_names(self) -> list:
        """Get all the feature group names in this database"""
        return list(self.model_data.keys())

    def _model_group_details(self, model_group_name: str) -> dict:
        """Internal: Do not call this method directly, use model_group_details() instead"""

        # Grab the Model Group details from the AWS Model Registry
        details = self.sm_client.list_model_packages(ModelPackageGroupName=model_group_name)["ModelPackageSummaryList"]
        for detail in details:
            model_package_arn = detail["ModelPackageArn"]
            detail["ModelPackageDetails"] = self.sm_client.describe_model_package(ModelPackageName=model_package_arn)
            detail["ModelPackageGroupArn"] = self.model_package_group_arns[model_group_name]
        return details


if __name__ == "__main__":
    from pprint import pprint

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print("Unrecognized args: %s" % commands)
        sys.exit(1)

    # Create the class and get the AWS Model Registry details
    model_registry = ModelRegistry()
    model_registry.refresh()

    # List the Model Groups
    print("Model Groups:")
    for my_group_name in model_registry.model_group_names():
        print(f"\t{my_group_name}")

    # Print out the metadata sizes for this connector
    pprint(model_registry.get_metadata_sizes())
