"""EndpointWebView pulls Endpoint metadata from the AWS Service Broker with Details Panels on each Endpoint"""
import pandas as pd

# SageWorks Imports
from sageworks.views.artifacts_web_view import ArtifactsWebView
from sageworks.artifacts.endpoints.endpoint import Endpoint


class EndpointWebView(ArtifactsWebView):
    def __init__(self):
        """EndpointWebView pulls Endpoint metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # DataFrame of the Endpoints Summary
        self.endpoint_df = self.endpoints_summary()

    def refresh(self):
        """Refresh the data from the AWS Service Broker"""
        super().refresh()
        self.endpoint_df = self.endpoints_summary()

    def view_data(self) -> pd.DataFrame:
        """Get all the data that's useful for this view

        Returns:
            pd.DataFrame: DataFrame of the Endpoints View Data
        """
        return self.endpoint_df

    def endpoint_details(self, endpoint_index: int) -> (dict, None):
        """Get all the details for the given Endpoint Index"""
        uuid = self.endpoint_name(endpoint_index)
        endpoint = Endpoint(uuid)
        return endpoint.details()

    def endpoint_name(self, endpoint_index: int) -> (str, None):
        """Helper method for getting the data source name for the given Endpoint Index"""
        if not self.endpoint_df.empty and endpoint_index < len(self.endpoint_df):
            data_uuid = self.endpoint_df.iloc[endpoint_index]["uuid"]
            return data_uuid
        else:
            return None


if __name__ == "__main__":
    # Exercising the EndpointWebView
    from pprint import pprint

    # Create the class and get the AWS Endpoint details
    endpoint_view = EndpointWebView()

    # List the Endpoints
    print("EndpointsSummary:")
    summary = endpoint_view.view_data()
    print(summary.head())

    # Get the details for the first Endpoint
    print("\nEndpointDetails:")
    details = endpoint_view.endpoint_details(0)
    pprint(details)
