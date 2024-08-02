"""EndpointWebView pulls Endpoint metadata from the AWS Service Broker with Details Panels on each Endpoint"""

import pandas as pd

# SageWorks Imports
from sageworks.web_views.artifacts_web_view import ArtifactsWebView
from sageworks.core.artifacts.endpoint_core import EndpointCore


class EndpointWebView(ArtifactsWebView):
    def __init__(self):
        """EndpointWebView pulls Endpoint metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # DataFrame of the Endpoints Summary
        self.endpoint_df = self.endpoints_summary()

    def refresh(self):
        """Refresh the data from the AWS Service Broker"""
        self.endpoint_df = self.endpoints_summary()

    def view_data(self) -> pd.DataFrame:
        """Get all the data that's useful for this view

        Returns:
            pd.DataFrame: DataFrame of the Endpoints View Data
        """
        return self.endpoint_df

    def endpoint_details(self, endpoint_uuid: str) -> (dict, None):
        """Get all the details for the given Endpoint UUID
         Args:
            endpoint_uuid(str): The UUID of the Endpoint
        Returns:
            dict: The details for the given Model (or None if not found)
        """
        endpoint = EndpointCore(endpoint_uuid)
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
    my_endpoint_uuid = summary["uuid"][0]
    print("\nEndpointDetails:")
    details = endpoint_view.endpoint_details(my_endpoint_uuid)
    pprint(details)

    # Give any broker threads time to finish
    time.sleep(1)
