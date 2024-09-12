"""DataSourceWebView pulls DataSource metadata from the AWS Service Broker with Details Panels on each DataSource"""

import pandas as pd

# SageWorks Imports
from sageworks.web_views.artifacts_web_view import ArtifactsWebView
from sageworks.api.data_source import DataSource


class DataSourceWebView(ArtifactsWebView):
    def __init__(self):
        """DataSourceWebView pulls DataSource metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # DataFrame of the DataSources Summary
        self.data_sources_df = self.data_sources_summary()

    def refresh(self):
        """Refresh the data from the AWS Service Broker"""
        self.data_sources_df = self.data_sources_summary()

    def view_data(self) -> pd.DataFrame:
        """Get all the data that's useful for this view

        Returns:
            pd.DataFrame: DataFrame of the DataSources View Data
        """
        return self.data_sources_df

    @staticmethod
    def data_source_smart_sample(data_uuid: str) -> pd.DataFrame:
        """Get a smart-sample dataframe (sample + outliers) for the given DataSource Index
        Args:
            data_uuid(str): The UUID of the DataSource
        Returns:
            pd.DataFrame: The smart-sample DataFrame
        """
        ds = DataSource(data_uuid)
        if not ds.exists():
            return pd.DataFrame({"uuid": [data_uuid], "status": ["NOT FOUND"]})
        if not ds.ready():
            status = ds.get_status()
            return pd.DataFrame({"uuid": [data_uuid], "status": [f"{status}"]})
        else:
            # Get the display columns
            display_columns = ds.view("display").columns
            smart_sample = ds.smart_sample()

            # Return the Smart Sample (with the display subset of the columns)
            return smart_sample[[col for col in display_columns if col in smart_sample.columns]]

    @staticmethod
    def data_source_details(data_uuid: str) -> (dict, None):
        """Get all the details for the given DataSource UUID
        Args:
            data_uuid(str): The UUID of the DataSource
        Returns:
            dict: The details for the given DataSource (or None if not found)
        """
        # Grab the DataSource, if it exists and is ready
        ds = DataSource(data_uuid)
        if not ds.exists() or not ds.ready():
            return None

        # Get the display columns
        display_columns = ds.view("display").columns

        # Subset some of the details using the display columns
        sub_details = ds.details()
        sub_details["column_stats"] = {
            k: sub_details["column_stats"][k] for k in display_columns if k in sub_details["column_stats"]
        }
        sub_details["column_details"] = {
            k: sub_details["column_details"][k] for k in display_columns if k in sub_details["column_details"]
        }

        # Subset the correlation details to avoid breaking the correlation matrix
        for column, data in sub_details["column_stats"].items():
            if "correlations" in data:
                data["correlations"] = {
                    k: data["correlations"][k] for k in display_columns if k in data["correlations"]
                }

        # Return the Subset of DataSource Details
        return sub_details


if __name__ == "__main__":
    # Exercising the DataSourceWebView
    import time
    from pprint import pprint

    # Create the class and get the AWS DataSource details
    data_view = DataSourceWebView()

    # List the DataSources
    print("DataSourcesSummary:")
    summary = data_view.view_data()
    print(summary.head())

    # Get the details for the first DataSource
    my_data_uuid = summary["uuid"][0]
    print(f"\nDataSourceDetails: {my_data_uuid}")
    details = data_view.data_source_details(my_data_uuid)
    pprint(details)

    # Get a sample dataframe for the given DataSources
    print(f"\nSampleDataFrame: {my_data_uuid}")
    sample_df = data_view.data_source_smart_sample(my_data_uuid)
    print(sample_df.shape)
    print(sample_df.head())

    # Give any broker threads time to finish
    time.sleep(1)
