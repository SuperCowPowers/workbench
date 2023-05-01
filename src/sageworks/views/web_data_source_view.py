"""WebDataSourceView pulls DataSource metadata from the AWS Service Broker with Details Panels on each DataSource"""
import sys
import argparse
import pandas as pd

# SageWorks Imports
from sageworks.views.view import View
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory
from sageworks.artifacts.data_sources.data_source import DataSource


class WebDataSourceView(View):
    def __init__(self):
        """WebDataSourceView pulls DataSource metadata from the AWS Service Broker with Details Panels on each DataSource"""
        # Call SuperClass Initialization
        super().__init__()

        # Get AWS Service information for the Data Catalog (DataSources)
        self.data_sources_meta = {}
        self.refresh()

        # Get a handle to the AWS Artifact Information class
        self.artifact_info = self.aws_broker.artifact_info

    def check(self) -> bool:
        """Can we connect to this view/service?"""
        return True  # I'm great, thx for asking

    def refresh(self):
        """Refresh data/metadata associated with this view"""
        data_catalog_info = self.aws_broker.get_metadata(ServiceCategory.DATA_CATALOG)
        # Just the sageworks database (not sagemaker_featurestore)
        self.data_sources_meta = data_catalog_info.get("sageworks", {})

    def view_data(self) -> dict:
        """Get all the data that's useful for this view

        Returns:
            dict: Dictionary of Pandas Dataframes, e.g. {'DATA_SOURCES': pd.DataFrame, ...}
        """
        return {"DATA_SOURCES": self.data_sources_summary()}  # Just the DataSources Summary Dataframe

    def data_sources_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks DataSources"""
        data_summary = []
        for name, info in self.data_sources_meta.items():
            # Get the size of the S3 Storage Object(s)
            size = self.artifact_info.s3_object_sizes(info["StorageDescriptor"]["Location"])
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

    def hyperlinks(self, name, detail_type):
        athena_url = f"https://{self.aws_account_clamp.region()}.console.aws.amazon.com/athena/home"
        link = f"<a href='{detail_type}' target='_blank'>{name}</a>"
        link += f" [<a href='{athena_url}' target='_blank'>query</a>]"
        return link

    def data_source_sample(self, data_source_index: int, max_rows=100) -> pd.DataFrame:
        """Get a sample dataframe for the given DataSource Index"""
        # Grab the a sample of N rows of the data source
        if self.data_sources_meta and data_source_index < len(self.data_sources_meta):
            data_uuid = list(self.data_sources_meta.keys())[data_source_index]
            sample_rows = DataSource(data_uuid).sample_df(max_rows=max_rows).head(max_rows)
        else:
            sample_rows = pd.DataFrame()
        return sample_rows

    def data_source_name(self, data_source_index: int) -> str:
        """Helper method for getting the data source name for the given DataSource Index"""
        if self.data_sources_meta and data_source_index < len(self.data_sources_meta):
            data_uuid = list(self.data_sources_meta.keys())[data_source_index]
            return data_uuid
        else:
            return "Unknown"

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

    # Create the class and get the AWS DataSource details
    data_view = WebDataSourceView()

    # List the DataSources
    print("DataSourcesSummary:")
    summary = data_view.view_data()["DATA_SOURCES"]
    print(summary.head())

    # Get a sample dataframe for the given DataSources
    sample_df = data_view.data_source_sample(0)
    print(sample_df.head())
