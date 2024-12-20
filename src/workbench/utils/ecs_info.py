import json
import os

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp


class ECSInfo:
    def __init__(self):
        # Fetch the ECS Metadata from a file that's mounted into the container
        self.ecs_metadata = self._fetch_ecs_metadata()

        # Spin up our AWSAccountClamp
        self.aws_account_clamp = AWSAccountClamp()
        self.boto3_session = self.aws_account_clamp.boto3_session
        self.ecs_client = self.boto3_session.client("ecs")
        self.elbv2_client = self.boto3_session.client("elbv2")

    def get_cluster_name(self):
        return self.ecs_metadata.get("Cluster", "Unknown")

    def get_service_name(self):
        containers = self.ecs_metadata.get("Containers", [])
        if containers:
            return containers[0].get("Labels", {}).get("com.amazonaws.ecs.service-name", "Unknown")
        return "Unknown"

    def _fetch_ecs_metadata(self):
        metadata_file_path = os.environ.get("ECS_CONTAINER_METADATA_FILE", "/etc/ecs_container_metadata.json")
        try:
            with open(metadata_file_path, "r") as f:
                metadata = json.load(f)
            return metadata
        except FileNotFoundError:
            print("Metadata file not found. Are you sure you're running this on ECS?")
            return {}

    def get_load_balancer_name(self):
        try:
            response = self.ecs_client.describe_services(
                cluster=self.get_cluster_name(), services=[self.get_service_name()]
            )

            # Extract Target Group ARN
            target_group_arn = None
            if "services" in response and len(response["services"]) > 0:
                if "loadBalancers" in response["services"][0] and len(response["services"][0]["loadBalancers"]) > 0:
                    target_group_arn = response["services"][0]["loadBalancers"][0]["targetGroupArn"]

            # Describe Load Balancers associated with the Target Group
            if target_group_arn:
                response = self.elbv2_client.describe_load_balancers()
                for lb in response["LoadBalancers"]:
                    load_balancer_arn = lb["LoadBalancerArn"]
                    load_balancer_name = "/".join(load_balancer_arn.split(":")[-1].split("/")[-3:])
                    return load_balancer_name

            # Return Unknown if we didn't find a load balancer
            return "Unknown"

        except Exception as e:
            print(f"Error: {e}")
            return "Unknown"


if __name__ == "__main__":
    """Exercise the DashboardMetrics class"""

    # Create the Class and query for metrics
    my_metrics = ECSInfo()
    print(f"Cluster Name: {my_metrics.get_cluster_name()}")
    print(f"Service Name: {my_metrics.get_service_name()}")
    print(f"Load Balancer Name: {my_metrics.get_load_balancer_name()}")
