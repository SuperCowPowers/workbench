"""ArtifactsTextView pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""
import sys
import argparse
import pandas as pd
from termcolor import colored

# SageWorks Imports
from sageworks.views.view import View
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory


class ArtifactsTextView(View):
    def __init__(self):
        """ArtifactsTextView pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""
        # Call SuperClass Initialization
        super().__init__()

        # Get AWS Service information for ALL the categories (data_source, feature_set, endpoints, etc)
        self.aws_artifact_data = self.aws_broker.get_all_metadata()

        # Get AWS Account Region
        self.aws_region = self.aws_account_clamp.region()

        # Setup Pandas output options
        pd.set_option("display.max_colwidth", 50)
        pd.set_option("display.max_columns", 15)
        pd.set_option("display.width", 1000)

    def check(self) -> bool:
        """Can we connect to this view/service?"""
        return True  # I'm great, thx for asking

    def refresh(self):
        """Refresh data/metadata associated with this view"""
        self.aws_artifact_data = self.aws_broker.get_all_metadata()

    def view_data(self) -> dict:
        """Get all the data that's useful for this view

        Returns:
            dict: Dictionary of Pandas Dataframes, e.g. {'INCOMING_DATA_S3': pd.DataFrame, ...}
        """

        # We're filling in Summary Data for all the AWS Services
        summary_data = {
            "INCOMING_DATA": self.incoming_data_summary(),
            "DATA_SOURCES": self.data_sources_summary(),
            "FEATURE_SETS": self.feature_sets_summary(),
            "MODELS": self.models_summary(),
            "ENDPOINTS": self.endpoints_summary(),
        }
        return summary_data

    @staticmethod
    def header_text(header_text: str) -> str:
        """Colorize text for the terminal"""
        color_map = {
            "INCOMING_DATA": "cyan",
            "DATA_SOURCES": "red",
            "FEATURE_SETS": "yellow",
            "MODELS": "green",
            "ENDPOINTS": "magenta",
        }
        header = f"\n{'='*111}\n{header_text}\n{'='*111}"
        return colored(header, color_map[header_text])

    def summary(self) -> None:
        """Give a summary of all the Artifact data to stdout"""
        for name, df in self.view_data().items():
            print(self.header_text(name))
            if df.empty:
                print("\tNo ArtifactsTextView Found")
            else:
                print(df.to_string(index=False))

    def incoming_data_summary(self) -> pd.DataFrame:
        """Get summary data about data in the incoming-data S3 Bucket"""
        data = self.aws_artifact_data[ServiceCategory.INCOMING_DATA_S3]
        data_summary = []
        for name, info in data.items():
            # Get the size of the S3 Storage Object(s)
            size = info.get("ContentLength") / 1_000_000
            summary = {
                "Name": "/".join(name.split("/")[-2:]).replace("incoming-data/", ""),
                "Size(MB)": f"{size:.2f}",
                "Modified": self.datetime_string(info.get("LastModified", "-")),
                "ContentType": str(info.get("ContentType", "-")),
                "ServerSideEncryption": info.get("ServerSideEncryption", "-"),
                "Tags": str(
                    info.get("tags", "-"),
                ),
            }
            data_summary.append(summary)

        # Make sure we have data else return just the column names
        if data_summary:
            return pd.DataFrame(data_summary)
        else:
            columns = ["Name", "Size(MB)", "Modified", "ContentType", "ServerSideEncryption", "Tags"]
            return pd.DataFrame(columns=columns)

    def data_sources_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks DataSources"""
        data_catalog = self.aws_artifact_data[ServiceCategory.DATA_CATALOG]
        data_summary = []

        # Get the SageWorks DataSources
        if "sageworks" in data_catalog:
            for name, info in data_catalog["sageworks"].items():  # Just the sageworks database
                # Get the size of the S3 Storage Object(s)
                size = self.aws_broker.get_s3_object_sizes(
                    ServiceCategory.DATA_SOURCES_S3, info["StorageDescriptor"]["Location"]
                )
                size = f"{size/1_000_000:.2f}"
                summary = {
                    "Name": name,
                    "Ver": info.get("VersionId", "-"),
                    "Size(MB)": size,
                    "Modified": self.datetime_string(info.get("UpdateTime")),
                    "Num Columns": self.num_columns(info),
                    "DataLake": info.get("IsRegisteredWithLakeFormation", "-"),
                    "Tags": info.get("Parameters", {}).get("sageworks_tags", "-"),
                    "Input": str(
                        info.get("Parameters", {}).get("sageworks_input", "-"),
                    ),
                }
                data_summary.append(summary)

        # Make sure we have data else return just the column names
        if data_summary:
            return pd.DataFrame(data_summary)
        else:
            columns = [
                "Name",
                "Ver",
                "Size(MB)",
                "Modified",
                "Num Columns",
                "DataLake",
                "Tags",
                "Input",
            ]
            return pd.DataFrame(columns=columns)

    def feature_sets_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks FeatureSets"""
        data = self.aws_artifact_data[ServiceCategory.FEATURE_STORE]
        data_summary = []
        for feature_group, group_info in data.items():
            # Get the SageWorks metadata for this Feature Group
            sageworks_meta = group_info.get("sageworks_meta", {})

            # Get the size of the S3 Storage Object(s)
            size = self.aws_broker.get_s3_object_sizes(
                ServiceCategory.FEATURE_SETS_S3, group_info["OfflineStoreConfig"]["S3StorageConfig"]["S3Uri"]
            )
            size = f"{size / 1_000_000:.2f}"
            summary = {
                "Feature Group": group_info["FeatureGroupName"],
                "Size(MB)": size,
                "Catalog DB": group_info["OfflineStoreConfig"].get("DataCatalogConfig", {}).get("Database", "-"),
                "Athena Table": group_info["OfflineStoreConfig"].get("DataCatalogConfig", {}).get("TableName", "-"),
                "Online": str(group_info.get("OnlineStoreConfig", {}).get("EnableOnlineStore", "False")),
                "Created": self.datetime_string(group_info.get("CreationTime")),
                "Tags": sageworks_meta.get("sageworks_tags", "-"),
                "Input": sageworks_meta.get("sageworks_input", "-"),
            }
            data_summary.append(summary)

        # Make sure we have data else return just the column names
        if data_summary:
            return pd.DataFrame(data_summary)
        else:
            columns = [
                "Feature Group",
                "Size(MB)",
                "Catalog DB",
                "Athena Table",
                "Online",
                "Created",
                "Tags",
                "Input",
            ]
            return pd.DataFrame(columns=columns)

    def models_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks Models"""
        data = self.aws_artifact_data[ServiceCategory.MODELS]
        model_summary = []
        for model_group_info, model_list in data.items():
            # Special Case for Model Groups without any Models
            if not model_list:
                summary = {"Model Group": model_group_info}
                model_summary.append(summary)
                continue

            # Get Summary information for each model in the model_list
            for model in model_list:
                # Get the SageWorks metadata for this Model Group
                sageworks_meta = model.get("sageworks_meta", {})
                summary = {
                    "Model Group": model["ModelPackageGroupName"],
                    "Ver": model["ModelPackageVersion"],
                    "Status": model["ModelPackageStatus"],
                    "Description": model["ModelPackageDescription"],
                    "Created": self.datetime_string(model.get("CreationTime")),
                    "Tags": sageworks_meta.get("sageworks_tags", "-"),
                    "Input": sageworks_meta.get("sageworks_input", "-"),
                }
                model_summary.append(summary)

        # Make sure we have data else return just the column names
        if model_summary:
            return pd.DataFrame(model_summary)
        else:
            columns = [
                "Model Group",
                "Ver",
                "Status",
                "Description",
                "Created",
                "Tags",
                "Input",
            ]
            return pd.DataFrame(columns=columns)

    def endpoints_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks Endpoints"""
        data = self.aws_artifact_data[ServiceCategory.ENDPOINTS]
        data_summary = []

        # Get Summary information for each endpoint
        for endpoint, endpoint_info in data.items():
            # Get the SageWorks metadata for this Endpoint
            sageworks_meta = endpoint_info.get("sageworks_meta", {})

            summary = {
                "Name": endpoint_info["EndpointName"],
                "Status": endpoint_info["EndpointStatus"],
                "Created": self.datetime_string(endpoint_info.get("CreationTime")),
                "DataCapture": str(endpoint_info.get("DataCaptureConfig", {}).get("EnableCapture", "False")),
                "Sampling(%)": str(endpoint_info.get("DataCaptureConfig", {}).get("CurrentSamplingPercentage", "-")),
                "Tags": sageworks_meta.get("sageworks_tags", "-"),
                "Input": sageworks_meta.get("sageworks_input", "-"),
            }
            data_summary.append(summary)

        # Make sure we have data else return just the column names
        if data_summary:
            return pd.DataFrame(data_summary)
        else:
            columns = [
                "Name",
                "Status",
                "Created",
                "DataCapture",
                "Sampling(%)",
                "Tags",
                "Input",
            ]
            return pd.DataFrame(columns=columns)

    @staticmethod
    def num_columns(data_info):
        """Helper: Compute the number of columns from the storage descriptor data"""
        try:
            return len(data_info["StorageDescriptor"]["Columns"])
        except KeyError:
            return "-"

    @staticmethod
    def datetime_string(datetime_obj):
        """Helper: Convert DateTime Object into a nice string"""
        if datetime_obj is None:
            return "-"
        # Date + Hour Minute
        return datetime_obj.strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    # Collect args from the command line
    parser = argparse.ArgumentParser()
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print("Unrecognized args: %s" % commands)
        sys.exit(1)

    # Create the class and get the AWS Model Registry details
    artifacts = ArtifactsTextView()

    # Pull the data for all Artifacts in the AWS Account
    artifacts.view_data()

    # Give a text summary of all the Artifacts in the AWS Account
    artifacts.summary()
