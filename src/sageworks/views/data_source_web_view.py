"""DataSourceWebView pulls DataSource metadata from the AWS Service Broker with Details Panels on each DataSource"""
import pandas as pd

# SageWorks Imports
from sageworks.views.artifacts_web_view import ArtifactsWebView
from sageworks.artifacts.data_sources.data_source import DataSource


class DataSourceWebView(ArtifactsWebView):
    def __init__(self):
        """DataSourceWebView pulls DataSource metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # DataFrame of the DataSources Summary
        self.data_sources_df = self.data_sources_summary()

    def refresh(self):
        """Refresh the data from the AWS Service Broker"""
        super().refresh()
        self.data_sources_df = self.data_sources_summary()

    def view_data(self) -> pd.DataFrame:
        """Get all the data that's useful for this view

        Returns:
            pd.DataFrame: DataFrame of the DataSources View Data
        """
        return self.data_sources_df

    def data_source_smart_sample(self, data_source_index: int) -> pd.DataFrame:
        """Get a smart-sample dataframe (sample + outliers) for the given DataSource Index"""
        uuid = self.data_source_name(data_source_index)
        ds = DataSource(uuid)
        if ds.ready():
            return ds.smart_sample()
        else:
            status = ds.get_status()
            return pd.DataFrame({"uuid": [uuid], "status": [f"{status}"]})

    def data_source_details(self, data_source_index: int) -> (dict, None):
        """Get all the details for the given DataSource Index"""
        data_uuid = self.data_source_name(data_source_index)
        ds = DataSource(data_uuid)
        if ds.ready():
            details_data = ds.details()
            details_data["column_stats"] = ds.column_stats()
            return details_data

        # If we get here, we couldn't get the details
        return None

    def data_source_name(self, data_source_index: int) -> (str, None):
        """Helper method for getting the data source name for the given DataSource Index"""
        if not self.data_sources_df.empty and data_source_index < len(self.data_sources_df):
            data_uuid = self.data_sources_df.iloc[data_source_index]["uuid"]
            return data_uuid
        else:
            return None


if __name__ == "__main__":
    # Exercising the DataSourceWebView
    from pprint import pprint

    # Create the class and get the AWS DataSource details
    data_view = DataSourceWebView()

    # List the DataSources
    print("DataSourcesSummary:")
    summary = data_view.view_data()
    print(summary.head())

    # Get the details for the first DataSource
    print("\nDataSourceDetails:")
    details = data_view.data_source_details(0)
    pprint(details)

    # Get a sample dataframe for the given DataSources
    print("\nSampleDataFrame:")
    sample_df = data_view.data_source_smart_sample(0)
    print(sample_df.shape)
    print(sample_df.head())
