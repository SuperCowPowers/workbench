"""ArtifactsTextView pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""

import pandas as pd
from typing import Dict

# SageWorks Imports
from sageworks.views.view import View
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory
from sageworks.utils.datetime_utils import datetime_string
from sageworks.utils.aws_utils import aws_url
from sageworks.api.meta import Meta


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

        # Create a Meta object to get the AWS Account Info
        self.meta = Meta()

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
        return self.meta.data_sources()

    def feature_sets_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks FeatureSets"""
        return self.meta.feature_sets()

    def models_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks Models"""
        return self.meta.models()

    def endpoints_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks Endpoints"""
        return self.meta.endpoints()


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
