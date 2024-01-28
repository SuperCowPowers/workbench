"""FeatureSetWebView pulls FeatureSet metadata from the AWS Service Broker with Details Panels on each FeatureSet"""

import pandas as pd

# SageWorks Imports
from sageworks.views.artifacts_web_view import ArtifactsWebView
from sageworks.core.artifacts.feature_set_core import FeatureSetCore


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

    def feature_set_smart_sample(self, feature_uuid: str) -> pd.DataFrame:
        """Get a smart-sample dataframe (sample + outliers) for the given FeatureSet Index
        Args:
            feature_uuid(str): The UUID of the DataSource
        Returns:
            pd.DataFrame: The smart-sample DataFrame
        """
        fs = FeatureSetCore(feature_uuid)
        if not fs.exists():
            return pd.DataFrame({"uuid": [feature_uuid], "status": ["NOT FOUND"]})
        if not fs.ready():
            status = fs.get_status()
            return pd.DataFrame({"uuid": [feature_uuid], "status": [f"{status}"]})
        else:
            # Grab the Smart Sample
            smart_sample = fs.smart_sample()

            # Return the Smart Sample (max 100 rows)
            return smart_sample[:100]

    @staticmethod
    def feature_set_details(feature_uuid: str) -> (dict, None):
        """Get all the details for the given FeatureSet UUID
        Args:
            feature_uuid(str): The UUID of the FeatureSet
        Returns:
            dict: The details for the given FeatureSet (or None if not found)
        """
        fs = FeatureSetCore(feature_uuid)
        if not fs.exists() or not fs.ready():
            return None

        # Return the FeatureSet Details
        return fs.details()


if __name__ == "__main__":
    # Exercising the FeatureSetWebView
    import time
    from pprint import pprint

    # Create the class and get the AWS FeatureSet details
    feature_view = FeatureSetWebView()

    # List the FeatureSets
    print("FeatureSetsSummary:")
    summary = feature_view.view_data()
    print(summary.head())

    # Get the details for the first FeatureSet
    my_feature_uuid = summary["uuid"][0]
    print("\nFeatureSetDetails:")
    details = feature_view.feature_set_details(my_feature_uuid)
    pprint(details)

    # Get a sample dataframe for the given FeatureSets
    print("\nSampleDataFrame:")
    sample_df = feature_view.feature_set_smart_sample(my_feature_uuid)
    print(sample_df.shape)
    print(sample_df.head())

    # Give any broker threads time to finish
    time.sleep(1)
