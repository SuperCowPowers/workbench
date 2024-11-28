"""AllMetaAsDataFrames pulls All the metadata from the Cloud Platform and organizes/summarizes it"""

import pandas as pd
from typing import Dict

# SageWorks Imports
from sageworks.web_interface.page_views.page_view import PageView
from sageworks.cached.cached_meta import CachedMeta
from sageworks.utils.deprecated_utils import hard_deprecated


@hard_deprecated
class AllMetaAsDataFrames(PageView):

    def __init__(self):
        """AllMetaAsDataFrames pulls All the metadata from the Cloud Platform and organizes/summarizes it"""
        # Call SuperClass Initialization
        super().__init__()

        # Setup Pandas output options
        pd.set_option("display.max_colwidth", 35)
        pd.set_option("display.width", 600)

        # Create a CachedMeta object to get all the Metadata for Artifacts
        self.meta = CachedMeta()

    def refresh(self, force_refresh: bool = False) -> None:
        """Refresh data/metadata associated with this view"""
        self.log.debug("We don't need to refresh anything for this view.")

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
            print(f"\n{name}")
            if df.empty:
                print("\tNo Artifacts Found")
            else:
                # Remove any columns that start with _
                df = df.loc[:, ~df.columns.str.startswith("_")]
                print(df.to_string(index=False))

    def incoming_data_summary(self) -> pd.DataFrame:
        """Get summary data about data in the incoming-data S3 Bucket"""
        return self.meta.incoming_data()

    def glue_jobs_summary(self) -> pd.DataFrame:
        """Get summary data about AWS Glue Jobs"""
        return self.meta.etl_jobs()

    def data_sources_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks DataSources"""
        return self.meta.data_sources()

    def feature_sets_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks FeatureSets"""
        return self.meta.feature_sets(details=True)

    def models_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks Models"""
        return self.meta.models(details=True)

    def endpoints_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks Endpoints"""
        return self.meta.endpoints()


if __name__ == "__main__":
    import time

    # Create the class and get the AWS Model Registry details
    artifacts = AllMetaAsDataFrames()

    # Pull the data for all Artifacts in the AWS Account
    artifacts.view_data()

    # Give a text summary of all the Artifacts in the AWS Account
    artifacts.summary()

    # Give any broker threads time to finish
    time.sleep(2)
