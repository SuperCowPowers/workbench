"""PipelinesPageView pulls Pipeline metadata from the AWS Service Broker with Details Panels on each Pipeline"""

import pandas as pd

# Workbench Imports
from workbench.web_interface.page_views.page_view import PageView
from workbench.cached.cached_meta import CachedMeta
from workbench.cached.cached_pipeline import CachedPipeline
from workbench.utils.symbols import tag_symbols


class PipelinesPageView(PageView):
    def __init__(self):
        """PipelinesPageView pulls Pipeline metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # CachedMeta object for Cloud Platform Metadata
        self.meta = CachedMeta()

        # Initialize the Pipelines DataFrame
        self.pipelines_df = None
        self.refresh()

    def refresh(self):
        """Refresh the pipeline data from the Cloud Platform"""
        self.log.important("Calling pipelines page view refresh()..")
        self.pipelines_df = self.meta.pipelines()

        # Drop the AWS URL column
        self.pipelines_df.drop(columns=["_aws_url"], inplace=True, errors="ignore")
        # Add Health Symbols to the Model Group Name
        if "Health" in self.pipelines_df.columns:
            self.pipelines_df["Health"] = self.pipelines_df["Health"].map(lambda x: tag_symbols(x))

    def pipelines(self) -> pd.DataFrame:
        """Get all the data that's useful for this view

        Returns:
            pd.DataFrame: DataFrame of the Pipelines View Data
        """
        return self.pipelines_df

    @staticmethod
    def pipeline_details(pipeline_uuid: str) -> (dict, None):
        """Get all the details for the given Pipeline UUID
         Args:
            pipeline_uuid(str): The UUID of the Pipeline
        Returns:
            dict: The details for the given Model (or None if not found)
        """
        pipeline = CachedPipeline(pipeline_uuid)
        if pipeline is None:
            return {"Status": "Not Found"}

        # Return the Pipeline Details
        return pipeline.details()


if __name__ == "__main__":
    # Exercising the PipelinesPageView
    import time
    from pprint import pprint

    # Create the class and get the AWS Pipeline details
    pipeline_view = PipelinesPageView()

    # List the Pipelines
    print("PipelinesSummary:")
    summary = pipeline_view.pipelines()
    print(summary.head())

    # Get the details for the first Pipeline
    my_pipeline_uuid = summary["Name"].iloc[0]
    print("\nPipelineDetails:")
    details = pipeline_view.pipeline_details(my_pipeline_uuid)
    pprint(details)

    # Give any broker threads time to finish
    time.sleep(1)
