"""EndpointsPageView pulls Endpoint metadata from the AWS Service Broker with Details Panels on each Endpoint"""

import pandas as pd

# Workbench Imports
from workbench.web_interface.page_views.page_view import PageView
from workbench.cached.cached_meta import CachedMeta
from workbench.cached.cached_endpoint import CachedEndpoint
from workbench.utils.symbols import tag_symbols


class EndpointsPageView(PageView):
    def __init__(self):
        """EndpointsPageView pulls Endpoint metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # CachedMeta object for Cloud Platform Metadata
        self.meta = CachedMeta()

        # Initialize the Endpoints DataFrame
        self.endpoints_df = None
        self.refresh()

    def refresh(self):
        """Refresh the endpoint data from the Cloud Platform"""
        self.log.important("Calling endpoint page view refresh()..")
        self.endpoints_df = self.meta.endpoints()

        # Drop the AWS URL column
        self.endpoints_df.drop(columns=["_aws_url"], inplace=True, errors="ignore")
        # Add Health Symbols to the Model Group Name
        if "Health" in self.endpoints_df.columns:
            self.endpoints_df["Health"] = self.endpoints_df["Health"].map(lambda x: tag_symbols(x))

    def endpoints(self) -> pd.DataFrame:
        """Get all the data that's useful for this view

        Returns:
            pd.DataFrame: DataFrame of the Endpoints View Data
        """
        return self.endpoints_df

    @staticmethod
    def endpoint_details(endpoint_uuid: str) -> (dict, None):
        """Get all the details for the given Endpoint UUID
         Args:
            endpoint_uuid(str): The UUID of the Endpoint
        Returns:
            dict: The details for the given Model (or None if not found)
        """
        endpoint = CachedEndpoint(endpoint_uuid)
        if not endpoint.exists():
            return {"Status": "Not Found"}
        elif not endpoint.ready():
            return {"health_tags": endpoint.get_health_tags()}

        # Return the Endpoint Details
        return endpoint.details()


if __name__ == "__main__":
    # Exercising the EndpointsPageView
    import time
    from pprint import pprint

    # Create the class and get the AWS Endpoint details
    endpoint_view = EndpointsPageView()

    # List the Endpoints
    print("EndpointsSummary:")
    summary = endpoint_view.endpoints()
    print(summary.head())

    # Get the details for the first Endpoint
    my_endpoint_uuid = summary["Name"].iloc[0]
    print("\nEndpointDetails:")
    details = endpoint_view.endpoint_details(my_endpoint_uuid)
    pprint(details)

    # Give any broker threads time to finish
    time.sleep(1)
