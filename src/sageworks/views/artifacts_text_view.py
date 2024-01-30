"""ArtifactsTextView pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict

# SageWorks Imports
from sageworks.views.view import View
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory
from sageworks.utils.iso_8601 import iso8601_to_datetime


class ArtifactsTextView(View):
    aws_artifact_data = None  # Class-level attribute for caching the AWS Artifact Data

    def __init__(self):
        """ArtifactsTextView pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""
        # Call SuperClass Initialization
        super().__init__()

        # Get AWS Service information for ALL the categories (data_source, feature_set, endpoints, etc)
        if ArtifactsTextView.aws_artifact_data is None:
            ArtifactsTextView.aws_artifact_data = self.aws_broker.get_all_metadata()

        # Setup Pandas output options
        pd.set_option("display.max_colwidth", 35)
        pd.set_option("display.width", 600)

    def refresh(self, force_refresh: bool = False) -> None:
        """Refresh data/metadata associated with this view"""
        ArtifactsTextView.aws_artifact_data = self.aws_broker.get_all_metadata(force_refresh=force_refresh)

    def view_data(self) -> Dict[str, pd.DataFrame]:
        """Get all the data that's useful for this view

        Returns:
            dict: Dictionary of Pandas Dataframes, e.g. {'INCOMING_DATA_S3': pd.DataFrame, ...}
        """

        # We're filling in Summary Data for all the AWS Services
        summary_data = {
            "INCOMING_DATA": self.incoming_data_summary(),
            "GLUE_JOBS": self.glue_jobs_summary(),
            "DATA_SOURCES": self.data_sources_summary(),
            "FEATURE_SETS": self.feature_sets_summary(),
            "MODELS": self.models_summary(),
            "ENDPOINTS": self.endpoints_summary(),
        }
        return summary_data

    def summary(self) -> None:
        """Give a summary of all the Artifact data to stdout"""
        for name, df in self.view_data().items():
            print(name)
            if df.empty:
                print("\tNo Artifacts Found")
            else:
                # Remove any columns that start with _
                df = df.loc[:, ~df.columns.str.startswith("_")]
                print(df.to_string(index=False))

    def incoming_data_summary(self) -> pd.DataFrame:
        """Get summary data about data in the incoming-data S3 Bucket"""
        data = ArtifactsTextView.aws_artifact_data[ServiceCategory.INCOMING_DATA_S3]
        if data is None:
            print("No incoming-data S3 Bucket Found! Please set your SAGEWORKS_BUCKET ENV var!")
            return pd.DataFrame(
                columns=["Name", "Size(MB)", "Modified", "ContentType", "ServerSideEncryption", "Tags", "_aws_url"]
            )
        data_summary = []
        for name, info in data.items():
            # Get the name and the size of the S3 Storage Object(s)
            name = "/".join(name.split("/")[-2:]).replace("incoming-data/", "")
            info["Name"] = name
            size = info.get("ContentLength") / 1_000_000
            summary = {
                "Name": name,
                "Size(MB)": f"{size:.2f}",
                "Modified": self.datetime_string(info.get("LastModified", "-")),
                "ContentType": str(info.get("ContentType", "-")),
                "ServerSideEncryption": info.get("ServerSideEncryption", "-"),
                "Tags": str(info.get("tags", "-")),
                "_aws_url": self.aws_url(info, "S3"),  # Hidden Column
            }
            data_summary.append(summary)

        # Make sure we have data else return just the column names
        if data_summary:
            return pd.DataFrame(data_summary)
        else:
            columns = [
                "Name",
                "Size(MB)",
                "Modified",
                "ContentType",
                "ServerSideEncryption",
                "Tags",
                "_aws_url",
            ]
            return pd.DataFrame(columns=columns)

    def glue_jobs_summary(self) -> pd.DataFrame:
        """Get summary data about AWS Glue Jobs"""
        glue_meta = ArtifactsTextView.aws_artifact_data[ServiceCategory.GLUE_JOBS]
        glue_summary = []

        # Get the information about each Glue Job
        for name, info in glue_meta.items():
            summary = {
                "Name": info["Name"],
                "GlueVersion": info["GlueVersion"],
                "Workers": info.get("NumberOfWorkers", "-"),
                "WorkerType": info.get("WorkerType", "-"),
                "Modified": self.datetime_string(info.get("LastModifiedOn")),
                "LastRun": self.datetime_string(info["sageworks_meta"]["last_run"]),
                "Status": info["sageworks_meta"]["status"],
                "_aws_url": self.aws_url(info, "GlueJob"),  # Hidden Column
            }
            glue_summary.append(summary)

        # Make sure we have data else return just the column names
        if glue_summary:
            return pd.DataFrame(glue_summary)
        else:
            columns = [
                "Name",
                "GlueVersion",
                "Workers",
                "WorkerType",
                "Modified",
                "LastRun",
                "Status",
                "_aws_url",
            ]
            return pd.DataFrame(columns=columns)

    def data_sources_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks DataSources"""
        data_catalog = ArtifactsTextView.aws_artifact_data[ServiceCategory.DATA_CATALOG]
        data_summary = []

        # Get the SageWorks DataSources
        if "sageworks" in data_catalog:
            for name, info in data_catalog["sageworks"].items():  # Just the sageworks database
                # Get the size of the S3 Storage Object(s)
                """Memory Tests
                size = self.aws_broker.get_s3_object_sizes(
                    ServiceCategory.DATA_SOURCES_S3,
                    info["StorageDescriptor"]["Location"],
                )
                size = f"{size/1_000_000:.2f}"
                """
                size = "-"
                summary = {
                    "Name": name,
                    "Size(MB)": size,
                    "Modified": self.datetime_string(info.get("UpdateTime")),
                    "Num Columns": self.num_columns_ds(info),
                    "DataLake": info.get("IsRegisteredWithLakeFormation", "-"),
                    "Tags": info.get("Parameters", {}).get("sageworks_tags", "-"),
                    "Input": str(
                        info.get("Parameters", {}).get("sageworks_input", "-"),
                    ),
                    "_aws_url": self.aws_url(info, "DataSource"),  # Hidden Column
                }
                data_summary.append(summary)

        # Make sure we have data else return just the column names
        if data_summary:
            return pd.DataFrame(data_summary)
        else:
            columns = [
                "Name",
                "Size(MB)",
                "Modified",
                "Num Columns",
                "DataLake",
                "Tags",
                "Input",
                "_aws_url",
            ]
            return pd.DataFrame(columns=columns)

    def feature_sets_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks FeatureSets"""
        data = ArtifactsTextView.aws_artifact_data[ServiceCategory.FEATURE_STORE]
        data_summary = []
        for feature_group, group_info in data.items():
            # Get the SageWorks metadata for this Feature Group
            sageworks_meta = group_info.get("sageworks_meta", {})

            # Get the size of the S3 Storage Object(s)
            """
            size = self.aws_broker.get_s3_object_sizes(
                ServiceCategory.FEATURE_SETS_S3,
                group_info["OfflineStoreConfig"]["S3StorageConfig"]["S3Uri"],
            )
            """
            size = 0
            size = f"{size / 1_000_000:.2f}"
            cat_config = group_info["OfflineStoreConfig"].get("DataCatalogConfig", {})
            summary = {
                "Feature Group": group_info["FeatureGroupName"],
                "Size(MB)": size,
                "Created": self.datetime_string(group_info.get("CreationTime")),
                "Num Columns": self.num_columns_fs(group_info),
                "Input": sageworks_meta.get("sageworks_input", "-"),
                "Tags": sageworks_meta.get("sageworks_tags", "-"),
                "Online": str(group_info.get("OnlineStoreConfig", {}).get("EnableOnlineStore", "False")),
                "Athena Table": cat_config.get("TableName", "-"),
                "_aws_url": self.aws_url(group_info, "FeatureSet"),  # Hidden Column
            }
            data_summary.append(summary)

        # Make sure we have data else return just the column names
        if data_summary:
            return pd.DataFrame(data_summary)
        else:
            columns = [
                "Feature Group",
                "Size(MB)",
                "Created",
                "Num Columns",
                "Input",
                "Tags",
                "Online",
                "Athena Table",
                "_aws_url",
            ]
            return pd.DataFrame(columns=columns)

    def models_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks Models"""
        data = ArtifactsTextView.aws_artifact_data[ServiceCategory.MODELS]
        model_summary = []
        for model_group_name, model_list in data.items():
            # Special Case for Model Groups without any Models
            if not model_list:
                summary = {
                    "Model Group": model_group_name,
                    "Health": "No Models!",
                    "Owner": "-",
                    "Model Type": None,
                    "Created": "-",
                    "Ver": "-",
                    "Tags": "-",
                    "Input": "-",
                    "Status": "Empty",
                    "Description": "-",
                }
                model_summary.append(summary)
                continue

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
                "Created": self.datetime_string(latest_model.get("CreationTime")),
                "Ver": latest_model["ModelPackageVersion"],
                "Tags": sageworks_meta.get("sageworks_tags", "-"),
                "Input": sageworks_meta.get("sageworks_input", "-"),
                "Status": latest_model["ModelPackageStatus"],
                "Description": latest_model.get("ModelPackageDescription", "-"),
            }
            model_summary.append(summary)

        # Make sure we have data else return just the column names
        if model_summary:
            return pd.DataFrame(model_summary)
        else:
            columns = [
                "Model Group",
                "Health",
                "Owner",
                "Model Type",
                "Created",
                "Ver",
                "Tags",
                "Input",
                "Status",
                "Description",
            ]
            return pd.DataFrame(columns=columns)

    def endpoints_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks Endpoints"""
        data = ArtifactsTextView.aws_artifact_data[ServiceCategory.ENDPOINTS]
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
                "Created": self.datetime_string(endpoint_info.get("CreationTime")),
                "Tags": sageworks_meta.get("sageworks_tags", "-"),
                "Input": sageworks_meta.get("sageworks_input", "-"),
                "Status": endpoint_info["EndpointStatus"],
                "Variant": endpoint_info.get("ProductionVariants", [{}])[0].get("VariantName", "-"),
                "Capture": str(endpoint_info.get("DataCaptureConfig", {}).get("EnableCapture", "False")),
                "Samp(%)": str(endpoint_info.get("DataCaptureConfig", {}).get("CurrentSamplingPercentage", "-")),
            }
            data_summary.append(summary)

        # Make sure we have data else return just the column names
        if data_summary:
            return pd.DataFrame(data_summary)
        else:
            columns = [
                "Name",
                "Health",
                "Instance",
                "Created",
                "Tags",
                "Input",
                "Status",
                "Variant",
                "Capture",
                "Samp(%)",
            ]
            return pd.DataFrame(columns=columns)

    @staticmethod
    def num_columns_ds(data_info):
        """Helper: Compute the number of columns from the storage descriptor data"""
        try:
            return len(data_info["StorageDescriptor"]["Columns"])
        except KeyError:
            return "-"

    @staticmethod
    def num_columns_fs(data_info):
        """Helper: Compute the number of columns from the feature group data"""
        try:
            return len(data_info["FeatureDefinitions"])
        except KeyError:
            return "-"

    def datetime_string(self, datetime_obj: datetime) -> str:
        """Helper: Convert DateTime Object into a nicely formatted string.

        Args:
            datetime_obj (datetime): The datetime object to convert.

        Returns:
            str: The datetime as a string in the format "YYYY-MM-DD HH:MM", or "-" if input is None or "-".
        """
        placeholder = "-"
        if datetime_obj is None or datetime_obj == placeholder:
            return placeholder

        if not isinstance(datetime_obj, datetime):
            self.log.debug("Expected datetime object.. trying to convert...")
            try:
                datetime_obj = iso8601_to_datetime(datetime_obj)
            except Exception as e:
                self.log.error(f"Failed to convert datetime object: {e}")
                return str(datetime_obj)

        try:
            return datetime_obj.strftime("%Y-%m-%d %H:%M")
        except Exception as e:
            self.log.error(f"Failed to convert datetime to string: {e}")
            return str(datetime_obj)

    def aws_url(self, artifact_info, artifact_type):
        """Helper: Try to extract the AWS URL from the Artifact Info Object"""
        if artifact_type == "S3":
            # Construct the AWS URL for the S3 Bucket
            name = artifact_info["Name"]
            region = self.aws_account_clamp.region
            s3_prefix = f"incoming-data/{name}"
            bucket_name = self.aws_account_clamp.sageworks_bucket_name
            base_url = "https://s3.console.aws.amazon.com/s3/object"
            return f"{base_url}/{bucket_name}?region={region}&prefix={s3_prefix}"
        elif artifact_type == "GlueJob":
            # Construct the AWS URL for the Glue Job
            region = self.aws_account_clamp.region
            job_name = artifact_info["Name"]
            base_url = f"https://{region}.console.aws.amazon.com/gluestudio/home"
            return f"{base_url}?region={region}#/editor/job/{job_name}/details"
        elif artifact_type == "DataSource":
            details = artifact_info.get("Parameters", {}).get("sageworks_details", "{}")
            return json.loads(details).get("aws_url", "unknown")
        elif artifact_type == "FeatureSet":
            aws_url = artifact_info.get("sageworks_meta", {}).get("aws_url", "unknown")
            # Hack for constraints on the SageMaker Feature Group Tags
            return aws_url.replace("__question__", "?").replace("__pound__", "#")


if __name__ == "__main__":
    import time

    # Create the class and get the AWS Model Registry details
    artifacts = ArtifactsTextView()

    # Pull the data for all Artifacts in the AWS Account
    artifacts.view_data()

    # Give a text summary of all the Artifacts in the AWS Account
    artifacts.summary()

    # Give any broker threads time to finish
    time.sleep(2)
