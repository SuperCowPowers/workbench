"""ArtifactsSummary pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""
import sys
import argparse
import awswrangler as wr
import pandas as pd

# SageWorks Imports
from sageworks.views.view import View
from sageworks.utils.cache import Cache
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp


class ArtifactsSummary(View):
    def __init__(self):
        """ArtifactsSummary pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""
        # Call SuperClass Initialization
        super().__init__()

        # Get AWS Service information for ALL the categories (data_source, feature_set, endpoints, etc)
        self.service_info = self.aws_broker.get_all_metadata()
        self.sm_session = AWSAccountClamp().sagemaker_session()

        # Summary data for ALL the AWS Services
        self.summary_data = {}

        # S3 Object Size Cache
        self.size_cache = Cache(expire=60)

    def check(self) -> bool:
        """Can we connect to this view/service?"""
        return True  # I'm great, thx for asking

    def refresh(self):
        """Refresh data/metadata associated with this view"""
        self.service_info = self.aws_broker.get_all_metadata()

    def s3_objects_size(self, s3_path) -> bool:
        """Return the size of this data in MegaBytes"""
        size_in_mb = self.size_cache.get(s3_path)
        if size_in_mb is None:
            self.log.info(f"Computing S3 Object sizes: {s3_path}...")
            size_in_bytes = sum(wr.s3.size_objects(s3_path, boto3_session=self.boto_session).values())
            size_in_mb = f"{ (size_in_bytes/1_000_000):.1f}"
            self.size_cache.set(s3_path, size_in_mb)
        return size_in_mb

    @staticmethod
    def aws_tags_to_dict(aws_tags):
        """AWS Tags are in an odd format, so convert to regular dictionary"""
        return {item["Key"]: item["Value"] for item in aws_tags if "sageworks" in item["Key"]}

    def artifact_meta(self, aws_arn):
        """Get the Metadata for this Artifact"""
        meta = self.size_cache.get(aws_arn)
        if meta is None:
            self.log.info(f"Retrieving Artifact Tags, Metadata, and S3 Object Size: {aws_arn}...")
            aws_tags = self.sm_session.list_tags(aws_arn)
            meta = self.aws_tags_to_dict(aws_tags)
            self.size_cache.set(aws_arn, meta)
        return meta

    def view_data(self) -> dict:
        """Get all the data that's useful for this view

        Returns:
            dict: Dictionary of Pandas Dataframes, i.e. {'INCOMING_DATA', pd.DataFrame}
        """

        # We're filling in Summary Data for all the AWS Services
        self.log.warning("Refreshing ALL AWS Service Broker Metadata...")
        self.summary_data["INCOMING_DATA"] = self.incoming_data_summary()
        self.summary_data["DATA_SOURCES"] = self.data_sources_summary()
        self.summary_data["FEATURE_SETS"] = self.feature_sets_summary()
        self.summary_data["MODELS"] = self.models_summary()
        self.summary_data["ENDPOINTS"] = self.endpoints_summary()
        return self.summary_data

    def incoming_data_summary(self):
        """Get summary data about data in the incoming-data S3 Bucket"""
        data = self.service_info[ServiceCategory.INCOMING_DATA]
        data_summary = []
        for name, info in data.items():
            # Get the size of the S3 Storage Object(s)
            size = info.get("ContentLength") / 1_000_000
            summary = {
                "Name": name,
                "Size(MB)": f"{size:.1f}",
                "Modified": self.datetime_string(info.get("LastModified", "-")),
                "ContentType": str(info.get("ContentType", "-")),
                "ServerSideEncryption": info.get("ServerSideEncryption", "-"),
                "Tags": str(
                    info.get("tags", "-"),
                ),
            }
            data_summary.append(summary)

        return pd.DataFrame(data_summary)

    def data_sources_summary(self):
        """Get summary data about the SageWorks DataSources"""
        data = self.service_info[ServiceCategory.DATA_CATALOG]
        data_summary = []
        for name, info in data["sageworks"].items():  # Just the sageworks database (not sagemaker_featurestore)
            # Get the size of the S3 Storage Object(s)
            size = self.s3_objects_size(info["StorageDescriptor"]["Location"])
            summary = {
                "Name": self.hyperlinks(name, "data_sources"),
                "Ver": info.get("VersionId", "-"),
                "Size(MB)": size,
                "Catalog DB": info.get("DatabaseName", "-"),
                # 'Created': self.datetime_string(info.get('CreateTime')),
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
                "Catalog DB",
                "Modified",
                "Num Columns",
                "DataLake",
                "Tags",
                "Input",
            ]
            return pd.DataFrame(columns=columns)

    @staticmethod
    def hyperlinks(name, detail_type):
        athena_url = "https://us-west-2.console.aws.amazon.com/athena/home"
        link = f"<a href='{detail_type}' target='_blank'>{name}</a>"
        link += f" [<a href='{athena_url}' target='_blank'>query</a>]"
        return link

    def feature_sets_summary(self):
        """Get summary data about the SageWorks FeatureSets"""
        data = self.service_info[ServiceCategory.FEATURE_STORE]
        data_summary = []
        for feature_group, group_info in data.items():
            # Get the tags for this Feature Group
            arn = group_info["FeatureGroupArn"]
            sageworks_meta = self.artifact_meta(arn)

            # Get the size of the S3 Storage Object(s)
            size = self.s3_objects_size(group_info["OfflineStoreConfig"]["S3StorageConfig"]["S3Uri"])
            summary = {
                "Feature Group": self.hyperlinks(group_info["FeatureGroupName"], "feature_sets"),
                # 'Status': group_info['FeatureGroupStatus'],
                "Size(MB)": size,
                "Catalog DB": group_info["OfflineStoreConfig"].get("DataCatalogConfig", {}).get("Database", "-"),
                "Athena Table": group_info["OfflineStoreConfig"].get("DataCatalogConfig", {}).get("TableName", "-"),
                # 'ID/EventTime': f"{group_info['RecordIdentifierFeatureName']}/{group_info['EventTimeFeatureName']}",
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

    def models_summary(self, concise=False):
        """Get summary data about the SageWorks Models"""
        data = self.service_info[ServiceCategory.MODELS]
        model_summary = []
        for model_group, model_list in data.items():
            # Special Case for Model Groups without any Models
            if not model_list:
                summary = {"Model Group": model_group}
                model_summary.append(summary)
                continue

            # Get Summary information for each model in the model_list
            for model in model_list:
                # Get the tags for this Model Group
                model_group_arn = model["ModelPackageGroupArn"]
                sageworks_meta = self.artifact_meta(model_group_arn)

                # Do they want the full details or just the concise summary?
                if concise:
                    summary = {
                        "Model Group": self.hyperlinks(model["ModelPackageGroupName"], "models"),
                        "Description": model["ModelPackageDescription"],
                        "Created": self.datetime_string(model.get("CreationTime")),
                        "Tags": sageworks_meta.get("sageworks_tags", "-"),
                        "Input": sageworks_meta.get("sageworks_input", "-"),
                    }
                else:
                    summary = {
                        "Model Group": self.hyperlinks(model["ModelPackageGroupName"], "models"),
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

    def endpoints_summary(self):
        """Get summary data about the SageWorks Endpoints"""
        data = self.service_info[ServiceCategory.ENDPOINTS]
        data_summary = []

        # Get Summary information for each endpoint
        for endpoint, endpoint_info in data.items():
            # Get the tags for this Model Group
            endpoint_arn = endpoint_info["EndpointArn"]
            sageworks_meta = self.artifact_meta(endpoint_arn)

            summary = {
                "Name": self.hyperlinks(endpoint_info["EndpointName"], "endpoints"),
                "Status": endpoint_info["EndpointStatus"],
                "Created": self.datetime_string(endpoint_info.get("CreationTime")),
                # "Modified": self.datetime_string(endpoint_info.get("LastModifiedTime")),
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
                "Modified",
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
    artifact_view = ArtifactsSummary()

    # List the Endpoint Names
    print("ArtifactsSummary:")
    for category, df in artifact_view.view_data().items():
        print(f"\n{category}")
        print(df.head())
