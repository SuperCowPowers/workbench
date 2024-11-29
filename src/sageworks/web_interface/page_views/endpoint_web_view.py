"""EndpointWebView pulls Endpoint metadata from the AWS Service Broker with Details Panels on each Endpoint"""

import pandas as pd

# SageWorks Imports
from sageworks.web_interface.page_views.page_view import PageView
from sageworks.cached.cached_meta import CachedMeta
from sageworks.cached.cached_endpoint import CachedEndpoint


class EndpointWebView(PageView):
    def __init__(self):
        """EndpointWebView pulls Endpoint metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # CachedMeta object for Cloud Platform Metadata
        self.meta = CachedMeta()
        self.endpoint_df = self.meta.endpoints()

    def refresh(self):
        """Refresh the endpoint data from the Cloud Platform"""
        self.log.important("Calling refresh()..")
        self.endpoint_df = self.meta.endpoints()

    def endpoints(self) -> pd.DataFrame:
        """Get all the data that's useful for this view

        Returns:
            pd.DataFrame: DataFrame of the Endpoints View Data
        """
        return self.endpoint_df

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
    # Exercising the EndpointWebView
    import time
    from pprint import pprint

    # Create the class and get the AWS Endpoint details
    endpoint_view = EndpointWebView()

    # List the Endpoints
    print("EndpointsSummary:")
    summary = endpoint_view.view_data()
    print(summary.head())

    # Get the details for the first Endpoint
    my_endpoint_uuid = summary["uuid"].iloc[0]
    print("\nEndpointDetails:")
    details = endpoint_view.endpoint_details(my_endpoint_uuid)
    pprint(details)

    # Give any broker threads time to finish
    time.sleep(1)
