"""MainPage pulls All the metadata from the Cloud Platform and organizes/summarizes it"""

import pandas as pd
from typing import Optional, Tuple


# Workbench Imports
from workbench.web_interface.page_views.page_view import PageView
from workbench.utils.symbols import tag_symbols
from workbench.cached.cached_meta import CachedMeta
from workbench.utils.pandas_utils import dataframe_delta


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

    def incoming_data_delta(self, previous_hash: str = None) -> Tuple[Optional[pd.DataFrame], str]:
        """Detect changes in the incoming data and return a new DataFrame if changed."""
        return dataframe_delta(self.incoming_data_summary, previous_hash)

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

    def glue_jobs_delta(self, previous_hash: str = None) -> Tuple[Optional[pd.DataFrame], str]:
        """Detect changes in the AWS Glue Jobs and return a new DataFrame if changed."""
        return dataframe_delta(self.glue_jobs_summary, previous_hash)

    def data_sources_summary(self) -> pd.DataFrame:
        """Get summary data about the Workbench DataSources

        Returns:
            pd.DataFrame: Summary data about the Workbench DataSources
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

    def data_sources_delta(self, previous_hash: str = None) -> Tuple[Optional[pd.DataFrame], str]:
        """Detect changes in the Workbench DataSources and return a new DataFrame if changed."""
        return dataframe_delta(self.data_sources_summary, previous_hash)

    def feature_sets_summary(self) -> pd.DataFrame:
        """Get summary data about the Workbench FeatureSets

        Returns:
            pd.DataFrame: Summary data about the Workbench FeatureSets
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

    def feature_sets_delta(self, previous_hash: str = None) -> tuple[Optional[pd.DataFrame], str]:
        """Detect changes in the Workbench FeatureSets and return a new DataFrame if changed."""
        return dataframe_delta(self.feature_sets_summary, previous_hash)

    def models_summary(self) -> pd.DataFrame:
        """Get summary data about the Workbench Models

        Returns:
            pd.DataFrame: Summary data about the Workbench Models
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

    def models_delta(self, previous_hash: str = None) -> Tuple[Optional[pd.DataFrame], str]:
        """Detect changes in the Workbench Models and return a new DataFrame if changed."""
        return dataframe_delta(self.models_summary, previous_hash)

    def endpoints_summary(self) -> pd.DataFrame:
        """Get summary data about the Workbench Endpoints

        Returns:
            pd.DataFrame: Summary data about the Workbench Endpoints
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

    def endpoints_delta(self, previous_hash: str = None) -> Tuple[Optional[pd.DataFrame], str]:
        """Detect changes in the Workbench Endpoints and return a new DataFrame if changed."""
        return dataframe_delta(self.endpoints_summary, previous_hash)


if __name__ == "__main__":
    import time

    # Create the class and get the AWS Model Registry details
    main_page = MainPage()

    # List all the artifacts in the main page
    print("All Cloud Platform Artifacts:")
    print("Incoming Data:")
    df = main_page.incoming_data_summary()
    print(df.head())
    print("\nGlue Jobs:")
    df = main_page.glue_jobs_summary()
    print(df.head())
    print("\nData Sources:")
    df = main_page.data_sources_summary()
    print(df.head())
    print("\nFeature Sets:")
    df = main_page.feature_sets_summary()
    print(df.head())
    print("\nModels:")
    df = main_page.models_summary()
    print(df.head())
    print("\nEndpoints:")
    df = main_page.endpoints_summary()
    print(df.head())

    # Test the Delta Functions
    print("Testing the Delta Functions:")
    print("Models Delta:")
    models, models_hash = main_page.models_delta()
    print(models)
    print(models_hash)
    model, models_hash = main_page.models_delta(models_hash)
    print(model)
    print(models_hash)

    # Give any broker threads time to finish
    time.sleep(1)
