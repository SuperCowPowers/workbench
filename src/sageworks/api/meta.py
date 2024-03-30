"""Meta: A class that provides high level information and summaries of SageWorks/AWS Artifacts.
The Meta class provides 'meta' information, what account are we in, what is the current
configuration, etc. It also provides summaries of the AWS Artifacts, such as Data Sources,
Feature Sets, Models, and Endpoints.
"""

import logging

import pandas as pd

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker, ServiceCategory
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.datetime_utils import datetime_string
from sageworks.utils.aws_utils import num_columns_ds, num_columns_fs, aws_url


class Meta:
    """Meta: A class that provides high level information and summaries of SageWorks/AWS Artifacts

    Common Usage:
    ```
    meta = Meta()
    meta.account()
    meta.config()
    meta.data_sources()
    ```
    """

    def __init__(self):
        """Meta Initialization"""
        self.log = logging.getLogger("sageworks")

        # Account and Service Brokers
        self.aws_account_clamp = AWSAccountClamp()
        self.aws_broker = AWSServiceBroker()
        self.cm = ConfigManager()

    def account(self) -> dict:
        """Print out the AWS Account Info

        Returns:
            dict: The AWS Account Info
        """
        return self.aws_account_clamp.get_aws_account_info()

    def config(self) -> dict:
        """Return the current SageWorks Configuration

        Returns:
            dict: The current SageWorks Configuration
        """
        return self.cm.get_all_config()

    def data_sources(self) -> pd.DataFrame:
        """Get a summary of the Data Sources in AWS

        Returns:
            pd.DataFrame: A summary of the Data Sources in AWS
        """
        data = self.data_sources_deep()
        data_summary = []

        # Pull in various bits of metadata for each data source
        for name, info in data.items():
            summary = {
                "Name": name,
                "Modified": datetime_string(info.get("UpdateTime")),
                "Num Columns": num_columns_ds(info),
                "Tags": info.get("Parameters", {}).get("sageworks_tags", "-"),
                "Input": str(
                    info.get("Parameters", {}).get("sageworks_input", "-"),
                ),
                "_aws_url": aws_url(info, "DataSource", self.aws_account_clamp),  # Hidden Column
            }
            data_summary.append(summary)

        # Return the summary
        return pd.DataFrame(data_summary)

    def data_sources_deep(self, refresh: bool = False, remove_sageworks_meta: bool = False) -> dict:
        """Get a deeper set of data for the Data Sources in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.
            remove_sageworks_meta (bool, optional): Remove the verbose SageWorks Metadata. Defaults to False.

        Returns:
            dict: A summary of the Data Sources in AWS
        """
        if refresh:
            self.aws_broker.refresh_aws_data(ServiceCategory.DATA_CATALOG)
        data = self.aws_broker.get_metadata(ServiceCategory.DATA_CATALOG)

        # Data Sources are in the 'sageworks' database
        data = data["sageworks"]

        # Remove the verbose SageWorks Metadata?
        if remove_sageworks_meta:
            data = self._remove_sageworks_meta(data)

        # Return the data
        return data

    def feature_sets(self, refresh: bool = False) -> pd.DataFrame:
        """Get a summary of the Feature Sets in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the Feature Sets in AWS
        """
        data = self.feature_sets_deep(refresh)
        data_summary = []

        # Pull in various bits of metadata for each feature set
        for name, group_info in data.items():
            sageworks_meta = group_info.get("sageworks_meta", {})
            summary = {
                "Feature Group": group_info["FeatureGroupName"],
                "Created": datetime_string(group_info.get("CreationTime")),
                "Num Columns": num_columns_fs(group_info),
                "Input": sageworks_meta.get("sageworks_input", "-"),
                "Tags": sageworks_meta.get("sageworks_tags", "-"),
                "Online": str(group_info.get("OnlineStoreConfig", {}).get("EnableOnlineStore", "False")),
                "_aws_url": aws_url(group_info, "FeatureSet", self.aws_account_clamp),  # Hidden Column
            }
            data_summary.append(summary)

        # Return the summary
        return pd.DataFrame(data_summary)

    def feature_sets_deep(self, refresh: bool = False) -> dict:
        """Get a deeper set of data for the Feature Sets in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Feature Sets in AWS
        """
        if refresh:
            self.aws_broker.refresh_aws_data(ServiceCategory.FEATURE_STORE)
        return self.aws_broker.get_metadata(ServiceCategory.FEATURE_STORE)

    def models(self, refresh: bool = False) -> pd.DataFrame:
        """Get a summary of the Models in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the Models in AWS
        """
        model_data = self.models_deep(refresh)
        model_summary = []
        for model_group_name, model_list in model_data.items():

            # Get Summary information for the 'latest' model in the model_list
            latest_model = model_list[0]
            sageworks_meta = latest_model.get("sageworks_meta", {})

            # If the sageworks_health_tags have nothing in them, then the model is healthy
            health_tags = sageworks_meta.get("sageworks_health_tags", "-")
            health_tags = health_tags if health_tags else "healthy"
            summary = {
                "Model Group": latest_model["ModelPackageGroupName"],
                "Health": health_tags,
                "Owner": sageworks_meta.get("sageworks_owner", "-"),
                "Model Type": sageworks_meta.get("sageworks_model_type"),
                "Created": datetime_string(latest_model.get("CreationTime")),
                "Ver": latest_model["ModelPackageVersion"],
                "Tags": sageworks_meta.get("sageworks_tags", "-"),
                "Input": sageworks_meta.get("sageworks_input", "-"),
                "Status": latest_model["ModelPackageStatus"],
                "Description": latest_model.get("ModelPackageDescription", "-"),
            }
            model_summary.append(summary)

        # Return the summary
        return pd.DataFrame(model_summary)

    def models_deep(self, refresh: bool = False) -> dict:
        """Get a deeper set of data for Models in AWS

         Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Models in AWS
        """
        if refresh:
            self.aws_broker.refresh_aws_data(ServiceCategory.MODELS)
        return self.aws_broker.get_metadata(ServiceCategory.MODELS)

    def endpoints(self, refresh: bool = False) -> pd.DataFrame:
        """Get a summary of the Endpoints in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the Endpoints in AWS
        """
        data = self.endpoints_deep(refresh)
        data_summary = []

        # Get Summary information for each endpoint
        for endpoint, endpoint_info in data.items():
            # Get the SageWorks metadata for this Endpoint
            sageworks_meta = endpoint_info.get("sageworks_meta", {})

            # If the sageworks_health_tags have nothing in them, then the endpoint is healthy
            health_tags = sageworks_meta.get("sageworks_health_tags", "-")
            health_tags = health_tags if health_tags else "healthy"
            summary = {
                "Name": endpoint_info["EndpointName"],
                "Health": health_tags,
                "Instance": endpoint_info.get("InstanceType", "-"),
                "Created": datetime_string(endpoint_info.get("CreationTime")),
                "Tags": sageworks_meta.get("sageworks_tags", "-"),
                "Input": sageworks_meta.get("sageworks_input", "-"),
                "Status": endpoint_info["EndpointStatus"],
                "Variant": endpoint_info.get("ProductionVariants", [{}])[0].get("VariantName", "-"),
                "Capture": str(endpoint_info.get("DataCaptureConfig", {}).get("EnableCapture", "False")),
                "Samp(%)": str(endpoint_info.get("DataCaptureConfig", {}).get("CurrentSamplingPercentage", "-")),
            }
            data_summary.append(summary)

        # Return the summary
        return pd.DataFrame(data_summary)

    def endpoints_deep(self, refresh: bool = False) -> dict:
        """Get a deeper set of data for Endpoints in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Endpoints in AWS
        """
        if refresh:
            self.aws_broker.refresh_aws_data(ServiceCategory.ENDPOINTS)
        return self.aws_broker.get_metadata(ServiceCategory.ENDPOINTS)

    def _remove_sageworks_meta(self, data: dict) -> dict:
        """Internal: Recursively remove any keys with 'sageworks_' in them"""

        # Recursively exclude any keys with 'sageworks_' in them
        summary_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                summary_data[key] = self._remove_sageworks_meta(value)
            elif not key.startswith("sageworks_"):
                summary_data[key] = value
        return summary_data


if __name__ == "__main__":
    """Exercise the SageWorks Meta Class"""
    from pprint import pprint

    # Create the class
    meta = Meta()

    # Get the AWS Account Info
    print("*** AWS Account ***")
    pprint(meta.account())

    # Get the SageWorks Configuration
    print("*** SageWorks Configuration ***")
    pprint(meta.config())

    # Get the Data Sources
    print("\n\n*** Data Sources ***")
    pprint(meta.data_sources())

    # Get the Feature Sets
    print("\n\n*** Feature Sets ***")
    pprint(meta.feature_sets())

    # Get the Models
    print("\n\n*** Models ***")
    pprint(meta.models())

    # Get the Endpoints
    print("\n\n*** Endpoints ***")
    pprint(meta.endpoints())

    # Now do a deep dive on all the Artifacts
    print("\n\n#")
    print("# Deep Dives ***")
    print("#")

    # Get the Data Sources
    print("\n\n*** Data Sources ***")
    pprint(meta.data_sources_deep())

    # Get the Feature Sets
    print("\n\n*** Feature Sets ***")
    pprint(meta.feature_sets_deep())

    # Get the Models
    print("\n\n*** Models ***")
    pprint(meta.models_deep())

    # Get the Endpoints
    print("\n\n*** Endpoints ***")
    pprint(meta.endpoints_deep())