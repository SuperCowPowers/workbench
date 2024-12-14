"""MainPage pulls All the metadata from the Cloud Platform and organizes/summarizes it"""

import pandas as pd

# SageWorks Imports
from sageworks.web_interface.page_views.page_view import PageView
from sageworks.utils.symbols import tag_symbols
from sageworks.cached.cached_meta import CachedMeta


class MainPage(PageView):
    def __init__(self):
        """MainPage pulls All the metadata from the Cloud Platform and organizes/summarizes it"""
        # Call SuperClass Initialization
        super().__init__()

        # CachedMeta object for Cloud Platform Metadata
        self.meta = CachedMeta()

    def refresh(self):
        """Refresh the data associated with this page view"""
        self.log.info("MainPage Refresh (does nothing)")

    def incoming_data_summary(self) -> pd.DataFrame:
        """Get summary data about the AWS Glue Jobs

        Returns:
            pd.DataFrame: Summary data about the AWS Glue Jobs
        """

        # We get the dataframe from our CachedMeta
        s3_data_df = self.meta.incoming_data()

        # We might get an empty dataframe
        if s3_data_df.empty:
            return s3_data_df

        # Add a UUID column
        s3_data_df["uuid"] = s3_data_df["Name"]

        # Drop the AWS URL column and return the dataframe
        s3_data_df.drop(columns=["_aws_url"], inplace=True, errors="ignore")
        return s3_data_df

    def glue_jobs_summary(self) -> pd.DataFrame:
        """Get summary data about the AWS Glue Jobs

        Returns:
            pd.DataFrame: Summary data about the AWS Glue Jobs
        """

        # We get the dataframe from our CachedMeta
        glue_df = self.meta.etl_jobs()

        # We might get an empty dataframe
        if glue_df.empty:
            return glue_df

        # Add a UUID column
        glue_df["uuid"] = glue_df["Name"]

        # Drop the AWS URL column and return the dataframe
        glue_df.drop(columns=["_aws_url"], inplace=True, errors="ignore")
        return glue_df

    def data_sources_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks DataSources

        Returns:
            pd.DataFrame: Summary data about the SageWorks DataSources
        """

        # We get the dataframe from our CachedMeta
        data_df = self.meta.data_sources()

        # We might get an empty dataframe
        if data_df.empty:
            return data_df

        # Add a UUID column
        data_df["uuid"] = data_df["Name"]

        # Drop the AWS URL column and return the dataframe
        data_df.drop(columns=["_aws_url"], inplace=True, errors="ignore")
        return data_df

    def feature_sets_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks FeatureSets

        Returns:
            pd.DataFrame: Summary data about the SageWorks FeatureSets
        """

        # We get the dataframe from our CachedMeta
        feature_df = self.meta.feature_sets(details=True)

        # We might get an empty dataframe
        if feature_df.empty:
            return feature_df

        # Add a UUID column
        feature_df["uuid"] = feature_df["Feature Group"]

        # Drop the AWS URL column and return the dataframe
        feature_df.drop(columns=["_aws_url"], inplace=True, errors="ignore")
        return feature_df

    def models_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks Models

        Returns:
            pd.DataFrame: Summary data about the SageWorks Models
        """

        # We get the dataframe from our CachedMeta and hyperlink the Model Group column
        model_df = self.meta.models(details=True)

        # We might get an empty dataframe
        if model_df.empty:
            return model_df

        # Add a UUID column
        model_df["uuid"] = model_df["Model Group"]

        # Drop some columns
        model_df.drop(columns=["Ver", "Status", "_aws_url"], inplace=True, errors="ignore")

        # Add Health Symbols to the Model Group Name
        if "Health" in model_df.columns:
            model_df["Health"] = model_df["Health"].map(lambda x: tag_symbols(x))

        return model_df

    def endpoints_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks Endpoints

        Returns:
            pd.DataFrame: Summary data about the SageWorks Endpoints
        """

        # We get the dataframe from our CachedMeta and hyperlink the Name column
        endpoint_df = self.meta.endpoints()

        # We might get an empty dataframe
        if endpoint_df.empty:
            return endpoint_df

        # Add a UUID column
        endpoint_df["uuid"] = endpoint_df["Name"]

        # Drop the AWS URL column
        endpoint_df.drop(columns=["_aws_url"], inplace=True, errors="ignore")

        # Add Health Symbols to the Endpoint Name
        if "Health" in endpoint_df.columns:
            endpoint_df["Health"] = endpoint_df["Health"].map(lambda x: tag_symbols(x))

        return endpoint_df


if __name__ == "__main__":
    import time

    # Create the class and get the AWS Model Registry details
    artifact_view = MainPage()

    # List all the artifacts in the main page
    print("All Cloud Platform Artifacts:")
    print("Incoming Data:")
    df = artifact_view.incoming_data_summary()
    print(df.head())
    print("\nGlue Jobs:")
    df = artifact_view.glue_jobs_summary()
    print(df.head())
    print("\nData Sources:")
    df = artifact_view.data_sources_summary()
    print(df.head())
    print("\nFeature Sets:")
    df = artifact_view.feature_sets_summary()
    print(df.head())
    print("\nModels:")
    df = artifact_view.models_summary()
    print(df.head())
    print("\nEndpoints:")
    df = artifact_view.endpoints_summary()
    print(df.head())

    # Give any broker threads time to finish
    time.sleep(1)
