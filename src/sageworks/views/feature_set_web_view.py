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

    def view_data(self) -> dict:
        """Get all the data that's useful for this view

        Returns:
            dict: Dictionary of Pandas Dataframes, e.g. {'DATA_SOURCES': pd.DataFrame, ...}
        """
        return {"DATA_SOURCES": self.feature_sets_df}  # Just the FeatureSets Summary Dataframe

    def feature_set_sample(self, feature_set_index: int) -> pd.DataFrame:
        """Get a sample dataframe for the given FeatureSet Index"""
        data_uuid = self.feature_set_name(feature_set_index)
        if data_uuid is not None:
            ds = FeatureSet(data_uuid)
            sample_rows = ds.sample_df()
        else:
            sample_rows = pd.DataFrame()
        return sample_rows

    def feature_set_quartiles(self, feature_set_index: int) -> (dict, None):
        """Get all columns quartiles for the given FeatureSet Index"""
        data_uuid = self.feature_set_name(feature_set_index)
        if data_uuid is not None:
            quartiles_data = FeatureSet(data_uuid).quartiles()
            return quartiles_data
        else:
            return None

    def feature_set_smart_sample(self, feature_set_index: int) -> pd.DataFrame:
        """Get a SMART sample dataframe for the given FeatureSet Index
        Note:
            SMART here means a sample data + quartiles for each column"""
        # Sample DataFrame
        sample_rows = self.feature_set_sample(feature_set_index)

        # Quartiles Data
        quartiles_data = self.feature_set_quartiles(feature_set_index)
        if quartiles_data is None:
            return sample_rows

        # Convert the Quartiles Data into a DataFrame
        quartiles_dict_list = dict()
        for col_name, quartiles in quartiles_data.items():
            quartiles_dict_list[col_name] = quartiles.values()
        quartiles_df = pd.DataFrame(quartiles_dict_list)

        # Combine the sample rows with the quartiles data
        return pd.concat([sample_rows, quartiles_df]).reset_index(drop=True)

    def feature_set_details(self, feature_set_index: int) -> (dict, None):
        """Get all of the details for the given FeatureSet Index"""
        data_uuid = self.feature_set_name(feature_set_index)
        if data_uuid is not None:
            ds = FeatureSet(data_uuid)
            details_data = ds.details()
            details_data["value_counts"] = ds.value_counts()
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
    data_view = FeatureSetWebView()

    # List the FeatureSets
    print("FeatureSetsSummary:")
    summary = data_view.view_data()["DATA_SOURCES"]
    print(summary.head())

    # Get the details for the first FeatureSet
    print("\nFeatureSetDetails:")
    details = data_view.feature_set_details(0)
    pprint(details)

    # Get a sample dataframe for the given FeatureSets
    print("\nSampleDataFrame:")
    sample_df = data_view.feature_set_smart_sample(0)
    print(sample_df.shape)
    print(sample_df.head())
