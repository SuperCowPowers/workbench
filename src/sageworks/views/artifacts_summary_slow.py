"""ArtifactsSummary pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""
import sys
import argparse

import pandas as pd

# SageWorks Imports
from sageworks.views.view import View
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory
from sageworks.artifacts.data_sources.athena_source import AthenaSource


class ArtifactsSummary(View):
    def __init__(self):
        """ArtifactsSummary pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""
        # Call SuperClass Initialization
        super().__init__()

        # Get AWS Service information for ALL the categories (data_source, feature_set, endpoints, etc)
        self.service_info = self.aws_broker.get_all_metadata()

        # Summary data for ALL the AWS Services
        self.summary_data = {}

    def check(self) -> bool:
        """Can we connect to this view/service?"""
        True  # I'm great, thx for asking

    def refresh(self) -> bool:
        """Refresh data/metadata associated with this view"""
        self.service_info = self.aws_broker.get_all_metadata()

    def view_data(self) -> dict:
        """Get all the data that's useful for this view

            Returns:
                dict: Dictionary of Pandas Dataframes, i.e. {'INCOMING_DATA', pd.DataFrame}
           """

        # We're filling in Summary Data for all the AWS Services
        self.summary_data[ServiceCategory.INCOMING_DATA] = self.incoming_data_summary()
        self.summary_data[ServiceCategory.DATA_CATALOG] = self.data_sources_summary()
        self.summary_data[ServiceCategory.FEATURE_STORE] = self.incoming_data_summary()
        self.summary_data[ServiceCategory.MODELS] = self.incoming_data_summary()
        self.summary_data[ServiceCategory.ENDPOINTS] = self.incoming_data_summary()
        return self.summary_data

    def incoming_data_summary(self):
        """Get summary data about data in the incoming-data S3 Bucket"""
        data = self.service_info[ServiceCategory.INCOMING_DATA]
        data_summary = []
        for name, info in data.items():
            summary = {'Name': name,
                       'Size': str(info.get('ContentLength', '-')),
                       'LastModified': str(info.get('LastModified', '-')),
                       'ContentType': str(info.get('ContentType', '-')),
                       'ServerSideEncryption': info.get('ServerSideEncryption', '-'),
                       'Tags': str(info.get('tags', '-'), )}
            data_summary.append(summary)

        return pd.DataFrame(data_summary)

    def data_sources_summary(self):
        """Get summary data about the SageWorks DataSources"""
        data = self.service_info[ServiceCategory.DATA_CATALOG]
        data_summary = []
        for database, db_info in data.items():
            for name, info in db_info.items():

                # We pull from the Concrete Class to get nice API for metadata
                athena_source = AthenaSource(name, database=info.get('DatabaseName', 'sageworks'))
                summary = {'Name': athena_source.uuid(),
                           'Database': athena_source.data_catalog_db,
                           'Size': '-',  # athena_source.size(),  # This takes TOO LONG
                           'NumColumns': athena_source.num_columns(),
                           'NumRows': '-',  # athena_source.num_rows(),  # This takes TOO LONG
                           'Created': athena_source.created(),
                           'LastModified': athena_source.modified(),
                           'Location': athena_source.s3_storage_location(),
                           'AmazonURL': athena_source.aws_url(),
                           'DataLake Registered': info.get('IsRegisteredWithLakeFormation', '-'),
                           'Tags': str(athena_source.tags())}
                data_summary.append(summary)

        return pd.DataFrame(data_summary)

    def feature_store(self):
        """Get summary data about the SageWorks DataSources"""

        # FIXME: Fill in with real logic
        return self.data_sources_summary()

    def models(self):
        """Get summary data about the SageWorks DataSources"""

        # FIXME: Fill in with real logic
        return self.data_sources_summary()

    def endpoints(self):
        """Get summary data about the SageWorks DataSources"""

        # FIXME: Fill in with real logic
        return self.data_sources_summary()


if __name__ == '__main__':

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print('Unrecognized args: %s' % commands)
        sys.exit(1)

    # Create the class and get the AWS Model Registry details
    artifact_view = ArtifactsSummary()

    # List the Endpoint Names
    print('ArtifactsSummary:')
    for category, df in artifact_view.view_data().items():
        print(f"\n{category}")
        print(df.head())
