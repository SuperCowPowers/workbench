"""Meta: A class that provides high level information and summaries of SageWorks/AWS Artifacts.
The Meta class provides 'meta' information, what account are we in, what is the current
configuration, etc. It also provides summaries of the AWS Artifacts, such as Data Sources,
Feature Sets, Models, and Endpoints.
"""

import logging

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker, ServiceCategory
from sageworks.utils.config_manager import ConfigManager


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
        self.aws_account = AWSAccountClamp()
        self.aws_broker = AWSServiceBroker()
        self.cm = ConfigManager()

    def account(self) -> dict:
        """Print out the AWS Account Info

        Returns:
            dict: The AWS Account Info
        """
        return self.aws_account.get_aws_account_info()

    def config(self) -> dict:
        """Return the current SageWorks Configuration

        Returns:
            dict: The current SageWorks Configuration
        """
        return self.cm.get_all_config()

    def data_sources(self, refresh: bool = False, include_sageworks_meta: bool = False) -> dict:
        """Get a summary of the Data Sources in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.
            include_sageworks_meta (bool, optional): Include the verbose SageWorks Metadata. Defaults to False.

        Returns:
            dict: A summary of the Data Sources in AWS
        """
        if refresh:
            self.aws_broker.refresh_aws_data(ServiceCategory.DATA_CATALOG)
        data = self.aws_broker.get_metadata(ServiceCategory.DATA_CATALOG)

        # Data Sources are in the 'sageworks' database
        data = data["sageworks"]

        # Include the verbose SageWorks Metadata?
        if include_sageworks_meta:
            return data

        # Remove the SageWorks Metadata
        data = self._remove_sageworks_meta(data)
        return data

    def feature_sets(self, refresh: bool = False) -> dict:
        """Get a summary of the Feature Sets in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Feature Sets in AWS
        """
        if refresh:
            self.aws_broker.refresh_aws_data(ServiceCategory.FEATURE_STORE)
        return self.aws_broker.get_metadata(ServiceCategory.FEATURE_STORE)

    def models(self, refresh: bool = False) -> dict:
        """Get a summary of the Models in AWS

         Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Models in AWS
        """
        if refresh:
            self.aws_broker.refresh_aws_data(ServiceCategory.MODELS)
        return self.aws_broker.get_metadata(ServiceCategory.MODELS)

    def endpoints(self, refresh: bool = False) -> dict:
        """Get a summary of the Endpoints in AWS

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
