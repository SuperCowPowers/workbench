"""Endpoints: Helper Class for AWS SageMaker Endpoints"""
import sys
import argparse
import botocore

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.connector import Connector
from sageworks.utils.boto_error_info import client_error_info


class Endpoints(Connector):
    def __init__(self):
        """Endpoints: Helper Class for AWS SageMaker Endpoints"""
        # Call SuperClass Initialization
        super().__init__()

        # Set up our internal data storage
        self.endpoint_data = {}

    def check(self) -> bool:
        """Check if we can reach/connect to this AWS Service"""
        try:
            self.sm_client.list_endpoints()
            return True
        except Exception as e:
            self.log.critical(f"Error connecting to AWS SageMaker Endpoints: {e}")
            return False

    def refresh_impl(self):
        """Grab all the Endpoint Data from SageMaker"""
        self.log.info("Reading Endpoints from SageMaker...")
        _endpoints = self.sm_client.list_endpoints(MaxResults=100)["Endpoints"]
        _end_names = [_endpoint["EndpointName"] for _endpoint in _endpoints]

        # Get the details for Endpoints and convert to a data structure with direct lookup
        self.endpoint_data = {name: self._retrieve_details(name) for name in _end_names}

        # Additional details under the sageworks_meta section for each Endpoint
        for _end_name, end_info in self.endpoint_data.items():
            sageworks_meta = self.sageworks_meta_via_arn(end_info["EndpointArn"])
            end_info["sageworks_meta"] = sageworks_meta
        self.log.info("Done with Endpoints...")

    def aws_meta(self) -> dict:
        """Get the full AWS metadata about endpoints"""
        return self.endpoint_data

    def endpoint_names(self) -> list:
        """Get all the endpoint names in AWS"""
        return list(self.endpoint_data.keys())

    def endpoint_details(self, endpoint_name: str) -> dict:
        """Get the details for a specific endpoint"""
        return self.endpoint_data.get(endpoint_name)

    def _retrieve_details(self, endpoint_name: str) -> dict:
        """Internal: Do not call this method directly, use endpoint_details() instead"""

        # Grab the Endpoint details from AWS
        details = self.sm_client.describe_endpoint(EndpointName=endpoint_name)

        # We just need the instance type from the Endpoint Config
        try:
            endpoint_config = self.sm_client.describe_endpoint_config(EndpointConfigName=details["EndpointConfigName"])
            instance_type = endpoint_config["ProductionVariants"][0].get("InstanceType")
            if instance_type is None:
                mem_size = endpoint_config["ProductionVariants"][0]["ServerlessConfig"]["MemorySizeInMB"]
                concurrency = endpoint_config["ProductionVariants"][0]["ServerlessConfig"]["MaxConcurrency"]
                mem_in_gb = int(mem_size / 1024)
                instance_type = f"Serverless ({mem_in_gb}GB/{concurrency})"
            details["InstanceType"] = instance_type
            return details
        except botocore.exceptions.ClientError as e:
            client_error_info(e)
            details["InstanceType"] = "No Endpoint Config"
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

    # Create the class and get the AWS Endpoint details
    my_endpoints = Endpoints()
    my_endpoints.refresh()

    # List the Endpoint Names and Details
    print("Endpoints:")
    for end_name in my_endpoints.endpoint_names():
        print(f"\t{end_name}")
        pprint(my_endpoints.endpoint_details(end_name))
