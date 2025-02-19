"""FeatureSetsPageView pulls FeatureSet metadata from the AWS Service Broker with Details Panels on each FeatureSet"""

import pandas as pd

# Workbench Imports
from workbench.web_interface.page_views.page_view import PageView
from workbench.cached.cached_meta import CachedMeta
from workbench.cached.cached_feature_set import CachedFeatureSet


class FeatureSetsPageView(PageView):
    def __init__(self):
        """FeatureSetsPageView pulls FeatureSet metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # CachedMeta object for Cloud Platform Metadata
        self.meta = CachedMeta()

        # Initialize the Endpoints DataFrame
        self.feature_sets_df = None
        self.refresh()

    def refresh(self):
        """Refresh the data from the AWS Service Broker"""
        self.log.important("Calling featuresets page view refresh()..")
        self.feature_sets_df = self.meta.feature_sets(details=True)

    def feature_sets(self) -> pd.DataFrame:
        """Get a list of all the Feature

        Returns:
            pd.DataFrame: DataFrame of all the FeatureSets
        """
        return self.feature_sets_df

    @staticmethod
    def feature_set_smart_sample(feature_uuid: str) -> pd.DataFrame:
        """Get a smart-sample dataframe (sample + outliers) for the given FeatureSet Index
        Args:
            feature_uuid(str): The UUID of the DataSource
        Returns:
            pd.DataFrame: The smart-sample DataFrame
        """
        fs = CachedFeatureSet(feature_uuid)
        if not fs.exists():
            return pd.DataFrame({"uuid": [feature_uuid], "status": ["NOT FOUND"]})
        if not fs.ready():
            status = fs.get_status()
            return pd.DataFrame({"uuid": [feature_uuid], "status": [f"{status}"]})
        else:
            # Get the display columns
            display_columns = fs.view("display").columns

            # Return the Smart Sample (with the display subset of the columns)
            smart_sample = fs.smart_sample()
            return smart_sample[[col for col in display_columns if col in smart_sample.columns]]

    @staticmethod
    def feature_set_details(feature_uuid: str) -> (dict, None):
        """Get all the details for the given FeatureSet UUID
        Args:
            feature_uuid(str): The UUID of the FeatureSet
        Returns:
            dict: The details for the given FeatureSet (or None if not found)
        """
        fs = CachedFeatureSet(feature_uuid)
        if not fs.exists() or not fs.ready():
            return None

        # Get the display columns
        display_columns = fs.view("display").columns

        # Subset some of the details using the display columns
        sub_details = fs.details()
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
    # Exercising the FeatureSetsPageView
    import time
    from pprint import pprint

    # Create the class and get the AWS FeatureSet details
    feature_view = FeatureSetsPageView()

    # List the FeatureSets
    print("FeatureSetsSummary:")
    summary = feature_view.feature_sets()
    print(summary.head())

    # Get the details for the first FeatureSet
    my_feature_uuid = summary["Feature Group"].iloc[0]
    print("\nFeatureSetDetails:")
    details = feature_view.feature_set_details(my_feature_uuid)
    pprint(details)

    # Get a sample dataframe for the given FeatureSets
    print("\nSampleDataFrame:")
    sample_df = feature_view.feature_set_smart_sample(my_feature_uuid)
    print(sample_df.shape)
    print(sample_df.head())

    # Test refresh
    print("\nRefreshing...")
    feature_view.refresh()

    # Give any broker threads time to finish
    time.sleep(1)
