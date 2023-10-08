"""ModelRegistry: Helper Class for the AWS Model Registry Service"""
import sys
import argparse

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.connector import Connector


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

    def refresh_impl(self):
        """Load/reload the tables in the database"""
        # Grab all the Model Groups in the AWS Model Registry
        _model_groups = self.sm_client.list_model_package_groups()["ModelPackageGroupSummaryList"]
        _mg_names = [model_group["ModelPackageGroupName"] for model_group in _model_groups]

        # Grab the ModelPackageGroupARNs (we'll use them later to store in the model_data)
        self.model_package_group_arns = {g["ModelPackageGroupName"]: g["ModelPackageGroupArn"] for g in _model_groups}

        # Get the details for each Model Group and convert to a data structure with direct lookup
        self.model_data = {name: self._model_group_details(name) for name in _mg_names}

        # Additional details under the sageworks_meta section for each Model Group
        for mg_name in _mg_names:
            sageworks_meta = self.sageworks_meta_via_arn(self.model_package_group_arns[mg_name])
            # Model groups have a list of models
            for model_info in self.model_data[mg_name]:
                model_info["sageworks_meta"] = sageworks_meta

    def aws_meta(self) -> dict:
        """Return ALL the AWS metadata for the AWS Model Registry"""
        return self.model_data

    def model_group_names(self) -> list:
        """Get all the feature group names in this database"""
        return list(self.model_data.keys())

    def model_group_details(self, model_group_name: str) -> dict:
        """Get the details for a specific feature group"""
        return self.model_data.get(model_group_name)

    def _model_group_details(self, model_group_name: str) -> dict:
        """Internal: Do not call this method directly, use model_group_details() instead"""

        # Grab the Model Group details from the AWS Model Registry
        details = self.sm_client.list_model_packages(ModelPackageGroupName=model_group_name)["ModelPackageSummaryList"]
        for detail in details:
            model_arn = detail["ModelPackageArn"]
            detail["ModelPackageDetails"] = self.sm_client.describe_model_package(ModelPackageName=model_arn)
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

    # Get the details for a specific Model Group
    my_group = "abalone-regression"
    group_info = model_registry.model_group_details(my_group)
    pprint(group_info)
