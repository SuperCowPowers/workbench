"""ArtifactsWebView pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""
import sys
import argparse
import pandas as pd

# SageWorks Imports
from sageworks.views.artifacts_text_view import ArtifactsTextView


class ArtifactsWebView(ArtifactsTextView):
    def __init__(self):
        """ArtifactsWebView pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""
        # Call SuperClass Initialization
        super().__init__()

    def data_sources_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks DataSources"""

        # We get the dataframe from the ArtifactsTextView and hyperlink the Name column
        data_df = super().data_sources_summary()
        data_df["uuid"] = data_df["Name"]
        data_df["Name"] = data_df["Name"].map(lambda x: self.hyperlinks(x, "data_sources"))
        return data_df

    def feature_sets_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks FeatureSets"""

        # We get the dataframe from the ArtifactsTextView and hyperlink the Feature Group column
        feature_df = super().feature_sets_summary()
        feature_df["uuid"] = feature_df["Feature Group"]
        feature_df["Feature Group"] = feature_df["Feature Group"].map(lambda x: self.hyperlinks(x, "feature_sets"))
        return feature_df

    def models_summary(self, concise=False) -> pd.DataFrame:
        """Get summary data about the SageWorks Models"""

        # We get the dataframe from the ArtifactsTextView and hyperlink the Model Group column
        model_df = super().models_summary()
        model_df["uuid"] = model_df["Model Group"]
        model_df["Model Group"] = model_df["Model Group"].map(lambda x: self.hyperlinks(x, "models"))
        return model_df

    def endpoints_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks Endpoints"""

        # We get the dataframe from the ArtifactsTextView and hyperlink the Name column
        endpoint_df = super().endpoints_summary()
        endpoint_df["uuid"] = endpoint_df["Name"]
        endpoint_df["Name"] = endpoint_df["Name"].map(lambda x: self.hyperlinks(x, "endpoints"))
        return endpoint_df

    def hyperlinks(self, name, detail_type):
        athena_url = f"https://{self.aws_region}.console.aws.amazon.com/athena/home"
        link = f"<a href='{detail_type}' target='_blank'>{name}</a>"
        link += f" [<a href='{athena_url}' target='_blank'>query</a>]"
        return link


if __name__ == "__main__":
    # Collect args from the command line
    parser = argparse.ArgumentParser()
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print("Unrecognized args: %s" % commands)
        sys.exit(1)

    # Create the class and get the AWS Model Registry details
    artifact_view = ArtifactsWebView()

    # List the Endpoint Names
    print("ArtifactsWebView:")
    for category, df in artifact_view.view_data().items():
        print(f"\n{category}")
        print(df.head())
