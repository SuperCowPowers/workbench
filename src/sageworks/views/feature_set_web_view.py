"""FeatureSetWebView pulls FeatureSet metadata from the AWS Service Broker with Details Panels on each FeatureSet"""
import pandas as pd

# SageWorks Imports
from sageworks.views.artifacts_web_view import ArtifactsWebView
from sageworks.artifacts.feature_sets.feature_set import FeatureSet


class FeatureSetWebView(ArtifactsWebView):
    def __init__(self):
        """FeatureSetWebView pulls FeatureSet metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # DataFrame of the FeatureSets Summary
        self.feature_sets_df = self.feature_sets_summary()

    def refresh(self):
        """Refresh the data from the AWS Service Broker"""
        super().refresh()
        self.feature_sets_df = self.feature_sets_summary()

    def view_data(self) -> pd.DataFrame:
        """Get all the data that's useful for this view

        Returns:
            pd.DataFrame: DataFrame of the FeatureSets View Data
        """
        return self.feature_sets_df

    def feature_set_outliers(self, feature_set_index: int) -> pd.DataFrame:
        """Get a dataframe of outliers for the given FeatureSet Index"""
        uuid = self.feature_set_name(feature_set_index)
        fs = FeatureSet(uuid)
        if fs.get_status() == "ready":
            return fs.outliers()
        else:
            status = fs.get_status()
            return pd.DataFrame({"uuid": [uuid], "status": [f"{status}"]})

    def feature_set_descriptive_stats(self, feature_set_index: int) -> (dict, None):
        """Get all columns descriptive stats for the given FeatureSet Index"""
        data_uuid = self.feature_set_name(feature_set_index)
        uuid = self.feature_set_name(feature_set_index)
        fs = FeatureSet(uuid)
        if fs.get_status() == "ready":
            return FeatureSet(data_uuid).descriptive_stats()
        else:
            return None

    def feature_set_smart_sample(self, feature_set_index: int) -> pd.DataFrame:
        """Get a SMART sample dataframe for the given FeatureSet Index
        Note:
            SMART here means a sample data + outliers for each column"""
        uuid = self.feature_set_name(feature_set_index)
        if uuid is not None:
            fs = FeatureSet(uuid)
            return fs.smart_sample()
        else:
            return pd.DataFrame()

    def feature_set_details(self, feature_set_index: int) -> (dict, None):
        """Get all the details for the given FeatureSet Index"""
        uuid = self.feature_set_name(feature_set_index)
        fs = FeatureSet(uuid)
        if fs.get_status() == "ready":
            details_data = fs.data_source.details()
            details_data["column_stats"] = fs.column_stats()
            return details_data
        else:
            return None

    def feature_set_name(self, feature_set_index: int) -> (str, None):
        """Helper method for getting the data source name for the given FeatureSet Index"""
        if not self.feature_sets_df.empty and feature_set_index < len(self.feature_sets_df):
            data_uuid = self.feature_sets_df.iloc[feature_set_index]["uuid"]
            return data_uuid
        else:
            return None


if __name__ == "__main__":
    # Exercising the FeatureSetWebView
    from pprint import pprint

    # Create the class and get the AWS FeatureSet details
    feature_view = FeatureSetWebView()

    # List the FeatureSets
    print("FeatureSetsSummary:")
    summary = feature_view.view_data()
    print(summary.head())

    # Get the details for the first FeatureSet
    print("\nFeatureSetDetails:")
    details = feature_view.feature_set_details(0)
    pprint(details)

    # Get a sample dataframe for the given FeatureSets
    print("\nSampleDataFrame:")
    sample_df = feature_view.feature_set_smart_sample(0)
    print(sample_df.shape)
    print(sample_df.head())
