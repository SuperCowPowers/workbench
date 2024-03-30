"""ArtifactsTextView pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""

import pandas as pd
from typing import Dict

# SageWorks Imports
from sageworks.views.view import View
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory
from sageworks.utils.datetime_utils import datetime_string
from sageworks.utils.aws_utils import num_columns_ds, num_columns_fs, aws_url


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
                "Modified": datetime_string(info.get("LastModified", "-")),
                "ContentType": str(info.get("ContentType", "-")),
                "ServerSideEncryption": info.get("ServerSideEncryption", "-"),
                "Tags": str(info.get("tags", "-")),
                "_aws_url": aws_url(info, "S3", self.aws_account_clamp),  # Hidden Column
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
                "Modified": datetime_string(info.get("LastModifiedOn")),
                "LastRun": datetime_string(info["sageworks_meta"]["last_run"]),
                "Status": info["sageworks_meta"]["status"],
                "_aws_url": aws_url(info, "GlueJob", self.aws_account_clamp),  # Hidden Column
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
                    "Modified": datetime_string(info.get("UpdateTime")),
                    "Num Columns": num_columns_ds(info),
                    "DataLake": info.get("IsRegisteredWithLakeFormation", "-"),
                    "Tags": info.get("Parameters", {}).get("sageworks_tags", "-"),
                    "Input": str(
                        info.get("Parameters", {}).get("sageworks_input", "-"),
                    ),
                    "_aws_url": aws_url(info, "DataSource", self.aws_account_clamp),  # Hidden Column
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
                "Created": datetime_string(group_info.get("CreationTime")),
                "Num Columns": num_columns_fs(group_info),
                "Input": sageworks_meta.get("sageworks_input", "-"),
                "Tags": sageworks_meta.get("sageworks_tags", "-"),
                "Online": str(group_info.get("OnlineStoreConfig", {}).get("EnableOnlineStore", "False")),
                "Athena Table": cat_config.get("TableName", "-"),
                "_aws_url": aws_url(group_info, "FeatureSet", self.aws_account_clamp),  # Hidden Column
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
                "Created": datetime_string(latest_model.get("CreationTime")),
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
                "Created": datetime_string(endpoint_info.get("CreationTime")),
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
