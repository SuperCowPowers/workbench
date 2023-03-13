"""ArtifactsSummary pulls All the metadata from the AWS Service Broker and organizes/summarizes it"""
import sys
import argparse

import pandas as pd

# SageWorks Imports
from sageworks.views.view import View
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory


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
        self.summary_data['INCOMING_DATA'] = self.incoming_data_summary()
        self.summary_data['DATA_SOURCES'] = self.data_sources_summary()
        self.summary_data['FEATURE_SETS'] = self.feature_sets_summary()
        self.summary_data['MODELS'] = self.models_summary()
        self.summary_data['ENDPOINTS'] = self.endpoints_summary()
        return self.summary_data

    def incoming_data_summary(self):
        """Get summary data about data in the incoming-data S3 Bucket"""
        data = self.service_info[ServiceCategory.INCOMING_DATA]
        data_summary = []
        for name, info in data.items():
            summary = {'Name': name,
                       'Size': str(info.get('ContentLength', '-')),
                       'LastModified': self.datetime_string(info.get('LastModified')),
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
                summary = {'Name': name,
                           'Catalog DB': info.get('DatabaseName', '-'),
                           'Size': str(info.get('ContentLength', '-')),
                           'Created': self.datetime_string(info.get('CreateTime')),
                           'LastModified': self.datetime_string(info.get('UpdateTime')),
                           'Num Columns': self.num_columns(info),
                           'DataLake': info.get('IsRegisteredWithLakeFormation', '-'),
                           'Tags': str(info.get('tags', '-'), )}
                data_summary.append(summary)

        return pd.DataFrame(data_summary)

    def feature_sets_summary(self):
        """Get summary data about the SageWorks FeatureSets"""
        data = self.service_info[ServiceCategory.FEATURE_STORE]
        data_summary = []
        for feature_group, group_info in data.items():
            summary = {'Feature Group': group_info['FeatureGroupName'],
                       'Catalog DB': group_info['OfflineStoreConfig']['DataCatalogConfig'].get('Database', '-'),
                       'Feature ID': group_info['RecordIdentifierFeatureName'],
                       'Feature EventTime': group_info['EventTimeFeatureName'],
                       'Online': str(group_info.get('OnlineStoreConfig', {}).get('EnableOnlineStore', 'False')),
                       'Created': self.datetime_string(group_info.get('CreationTime')),
                       'LastModified': self.datetime_string(group_info.get('LastModified')),
                       'Tags': str(group_info.get('tags', '-'), )}
            data_summary.append(summary)

        return pd.DataFrame(data_summary)

    def models_summary(self):
        """Get summary data about the SageWorks Models"""
        data = self.service_info[ServiceCategory.MODELS]
        data_summary = []
        for model_group, model_list in data.items():
            # Special Case for Model Groups without any Models
            if not model_list:
                summary = {'Model Group': model_group}
                data_summary.append(summary)
                continue

            # Get Summary information for each model in the model_list
            for model in model_list:
                summary = {'Model Group': model['ModelPackageGroupName'],
                           'Version': model['ModelPackageVersion'],
                           'Description': model['ModelPackageDescription'],
                           'Created': self.datetime_string(model.get('CreationTime')),
                           'LastModified': self.datetime_string(model.get('LastModifiedTime')),
                           'Status': str(model.get('ModelApprovalStatus', ' - ')),
                           'Tags': str(model.get('tags', '-'), )}
                data_summary.append(summary)

        return pd.DataFrame(data_summary)

    def endpoints_summary(self):
        """Get summary data about the SageWorks Endpoints"""
        data = self.service_info[ServiceCategory.ENDPOINTS]
        data_summary = []

        # Get Summary information for each endpoint
        for endpoint, endpoint_info in data.items():
            summary = {'Name': endpoint_info['EndpointName'],
                       'Status': endpoint_info['EndpointStatus'],
                       'Description': 'TBD',
                       'Created': self.datetime_string(endpoint_info.get('CreationTime')),
                       'LastModified': self.datetime_string(endpoint_info.get('LastModifiedTime')),
                       'DataCapture': str(endpoint_info.get('DataCaptureConfig', {}).get('EnableCapture', 'False')),
                       'SamplingPercent': str(endpoint_info.get('DataCaptureConfig', {}).get('CurrentSamplingPercentage', '-')),
                       'Tags': str(endpoint_info.get('tags', '-'), )}
            data_summary.append(summary)

        return pd.DataFrame(data_summary)


    @staticmethod
    def num_columns(data_info):
        """Helper: Compute the number of columns from the storage descriptor data"""
        try:
            return len(data_info['StorageDescriptor']['Columns'])
        except KeyError:
            return '-'

    @staticmethod
    def datetime_string(datetime_obj):
        """Helper: Convert DateTime Object into a nice string"""
        if datetime_obj is None:
            return '-'
        # Date + Hour Minute
        return datetime_obj.strftime("%Y-%m-%d %H:%M")


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
