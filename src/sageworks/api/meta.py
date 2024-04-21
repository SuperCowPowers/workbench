"""Meta: A class that provides high level information and summaries of SageWorks/AWS Artifacts.
The Meta class provides 'meta' information, what account are we in, what is the current
configuration, etc. It also provides metadata for AWS Artifacts, such as Data Sources,
Feature Sets, Models, and Endpoints.
"""

import logging
from typing import Union
import pandas as pd

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker, ServiceCategory
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.datetime_utils import datetime_string
from sageworks.utils.aws_utils import num_columns_ds, num_columns_fs, aws_url
from sageworks.api.pipeline_manager import PipelineManager


class Meta:
    """Meta: A class that provides Metadata for a broad set of AWS Artifacts

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

        # Pipeline Manager
        self.pipeline_manager = PipelineManager()

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

    def incoming_data(self) -> pd.DataFrame:
        """Get summary data about data in the incoming-data S3 Bucket

        Returns:
            pd.DataFrame: A summary of the data in the incoming-data S3 Bucket
        """
        data = self.incoming_data_deep()
        data_summary = []
        for name, info in data.items():
            # Get the name and the size of the S3 Storage Object(s)
            name = "/".join(name.split("/")[-2:]).replace("incoming-data/", "")
            info["Name"] = name
            size = info.get("ContentLength") / 1_000_000
            summary = {
                "Name": name,
                "Size(MB)": f"{size:.2f}",
                "Modified": datetime_string(info.get("LastModified", "-")),
                "ContentType": str(info.get("ContentType", "-")),
                "ServerSideEncryption": info.get("ServerSideEncryption", "-"),
                "Tags": str(info.get("tags", "-")),
                "_aws_url": aws_url(info, "S3", self.aws_account_clamp),  # Hidden Column
            }
            data_summary.append(summary)

        # Return the summary
        return pd.DataFrame(data_summary)

    def incoming_data_deep(self, refresh: bool = False) -> dict:
        """Get a deeper set of data for the Incoming Data in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Incoming Data in AWS
        """
        return self.aws_broker.get_metadata(ServiceCategory.INCOMING_DATA_S3, force_refresh=refresh)

    def glue_jobs(self) -> pd.DataFrame:
        """Get summary data about AWS Glue Jobs"""
        glue_meta = self.glue_jobs_deep()
        glue_summary = []

        # Get the information about each Glue Job
        for name, info in glue_meta.items():
            summary = {
                "Name": info["Name"],
                "GlueVersion": info["GlueVersion"],
                "Workers": info.get("NumberOfWorkers", "-"),
                "WorkerType": info.get("WorkerType", "-"),
                "Modified": datetime_string(info.get("LastModifiedOn")),
                "LastRun": datetime_string(info["sageworks_meta"]["last_run"]),
                "Status": info["sageworks_meta"]["status"],
                "_aws_url": aws_url(info, "GlueJob", self.aws_account_clamp),  # Hidden Column
            }
            glue_summary.append(summary)

        # Return the summary
        return pd.DataFrame(glue_summary)

    def glue_jobs_deep(self, refresh: bool = False) -> dict:
        """Get a deeper set of data for the Glue Jobs in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Glue Jobs in AWS
        """
        return self.aws_broker.get_metadata(ServiceCategory.GLUE_JOBS, force_refresh=refresh)

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

    def data_source_details(
        self, data_source_name: str, database: str = "sageworks", refresh: bool = False
    ) -> Union[dict, None]:
        """Get detailed information about a specific data source in AWS

        Args:
            data_source_name (str): The name of the data source
            database (str, optional): Glue database. Defaults to 'sageworks'.
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: Detailed information about the data source (or None if not found)
        """
        data = self.data_sources_deep(database=database, refresh=refresh)
        return data.get(data_source_name)

    def data_sources_deep(self, database: str = "sageworks", refresh: bool = False) -> dict:
        """Get a deeper set of data for the Data Sources in AWS

        Args:
            database (str, optional): Glue database. Defaults to 'sageworks'.
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Data Sources in AWS
        """
        data = self.aws_broker.get_metadata(ServiceCategory.DATA_CATALOG, force_refresh=refresh)

        # Data Sources are in two databases, 'sageworks' and 'sagemaker_featurestore'
        data = data[database]

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

    def feature_set_details(self, feature_set_name: str) -> dict:
        """Get detailed information about a specific feature set in AWS

        Args:
            feature_set_name (str): The name of the feature set

        Returns:
            dict: Detailed information about the feature set
        """
        data = self.feature_sets_deep()
        return data.get(feature_set_name, {})

    def feature_sets_deep(self, refresh: bool = False) -> dict:
        """Get a deeper set of data for the Feature Sets in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Feature Sets in AWS
        """
        return self.aws_broker.get_metadata(ServiceCategory.FEATURE_STORE, force_refresh=refresh)

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

    def model_details(self, model_group_name: str) -> dict:
        """Get detailed information about a specific model group in AWS

        Args:
            model_group_name (str): The name of the model group

        Returns:
            dict: Detailed information about the model group
        """
        data = self.models_deep()
        return data.get(model_group_name, {})

    def models_deep(self, refresh: bool = False) -> dict:
        """Get a deeper set of data for Models in AWS

         Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Models in AWS
        """
        return self.aws_broker.get_metadata(ServiceCategory.MODELS, force_refresh=refresh)

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
        return self.aws_broker.get_metadata(ServiceCategory.ENDPOINTS, force_refresh=refresh)

    def pipelines(self, refresh: bool = False) -> pd.DataFrame:
        """Get a summary of the SageWorks Pipelines

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the SageWorks Pipelines
        """
        data = self.pipeline_manager.list_pipelines()

        # Return the pipelines summary as a DataFrame
        return pd.DataFrame(data)

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

    def refresh_all_aws_meta(self) -> None:
        """Force a refresh of all the metadata"""
        self.aws_broker.get_all_metadata(force_refresh=True)


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

    # Get the Incoming Data
    print("\n\n*** Incoming Data ***")
    pprint(meta.incoming_data())

    # Get the Glue Jobs
    print("\n\n*** Glue Jobs ***")
    pprint(meta.glue_jobs())

    # Get the Data Sources
    print("\n\n*** Data Sources ***")
    pprint(meta.data_sources())

    # Get the Data Source Details
    print("\n\n*** Data Source Details ***")
    pprint(meta.data_source_details("abalone_data"))

    # Get the Feature Sets
    print("\n\n*** Feature Sets ***")
    pprint(meta.feature_sets())

    # Get the Models
    print("\n\n*** Models ***")
    pprint(meta.models())

    # Get the Endpoints
    print("\n\n*** Endpoints ***")
    pprint(meta.endpoints())

    # Get the Pipelines
    print("\n\n*** Pipelines ***")
    pprint(meta.pipelines())

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
