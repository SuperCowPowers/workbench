"""DFToFeatureSet: Class to publish a Pandas DataFrame into a FeatureSet (Athena/FeatureStore)"""
from datetime import datetime
import logging
import pandas as pd
import time
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker import get_execution_role

# Local imports
from sageworks.utils.logging import logging_setup
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.aws_artifacts.feature_sets.feature_set import FeatureSet
from sageworks.aws_service_broker.aws_iam_manager import AWSIAMManager

# Setup Logging
logging_setup()

# Good Reference for AWS FeatureStore functionality
# - https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-featurestore/feature_store_introduction.html
# AWS ROLES
# IMPORTANT: You must attach the following policies to your execution role: * AmazonS3FullAccess * AmazonSageMakerFeatureStoreAccess


class DFToFeatureSet(Transform):
    def __init__(self):
        """DFToFeatureSet: Class to publish a Pandas DataFrame into a FeatureSet (Athena/FeatureStore)"""

        # Call superclass init
        super().__init__()

        # Set up all my class instance vars
        self.log = logging.getLogger(__name__)
        self.input_df = None
        self.output_uuid = None
        self.data_catalog_db = 'sageworks'
        self.feature_set_s3_path = 's3://sageworks-feature-sets'
        self.id_column = None
        self.event_time_column = None
        self.sagemaker_session = Session()
        # sagemaker_role = get_execution_role('sagemaker')
        # print(sagemaker_role)
        # self.execution_role = get_execution_role()

    def input_type(self) -> TransformInput:
        """What Input Type does this Transform Consume"""
        return TransformInput.PANDAS_DF

    def output_type(self) -> TransformOutput:
        """What Output Type does this Transform Produce"""
        return TransformOutput.FEATURE_SET

    def set_input(self, input_df: pd.DataFrame, id_column, event_time_column=None):
        """Set the Input DataFrame for this Transform"""
        self.id_column = id_column
        self.event_time_column = event_time_column
        self.input_df = input_df

    def set_input_uuid(self, input_uuid: str):
        """Not Implemented: Just satisfying the Transform abstract method requirements"""
        pass

    def set_output_uuid(self, output_uuid: str):
        """Set the Name for the output Data Source"""
        self.output_uuid = output_uuid

    def get_output(self) -> FeatureSet:
        """Get the FeatureSet Output from this Transform"""
        return FeatureSet(self.output_uuid)

    def validate_input(self) -> bool:
        """Validate the Input for this Transform"""

        # Simple check on the input dataframe
        return self.input_df is not None and not self.input_df.empty

    def _ensure_event_time(self):
        """Internal: AWS Feature Store requires an EventTime field in all data stored"""
        if self.event_time_column is None or self.event_time_column not in self.input_df.columns:
            current_datetime = datetime.utcnow()
            self.log.info('Generating an EventTime column before FeatureSet Creation...')
            self.event_time_column = 'EventTime'
            self.input_df[self.event_time_column] = pd.Series([current_time_sec] * len(self.input_df), dtype="float64")
        else:  # The event_time_column is defined so lets make sure it the right type for Feature Store
            if pd.api.types.is_datetime64_dtype(self.input_df[self.event_time_column]):
                self.log.info(f"Converting {self.event_time_column} to ISOFormat Date String before FeatureSet Creation...")

                # Convert the datetime Dtype to ISO-8601 string
                # Note: We assume the that datetimes are in UTC
                self.input_df[self.event_time_column] = self.input_df[self.event_time_column].map(lambda x: x.isoformat())

    def _convert_objs_to_string(self):
        """Internal: AWS Feature Store doesn't know how to store object dtypes, so convert to String"""
        for col in self.input_df:
            if pd.api.types.is_object_dtype(self.input_df[col].dtype):
                self.input_df[col] = self.input_df[col].astype(pd.StringDtype())

    def transform(self, overwrite: bool = True):
        """Convert the Pandas DataFrame into Parquet Format in the SageWorks S3 Bucket, and
           store the information about the data to the AWS Data Catalog sageworks database"""

        # Ensure that the input dataframe has an EventTime field
        self._ensure_event_time()

        # Convert object dtypes to string
        self._convert_objs_to_string()

        # Create a Feature Group and load our Feature Definitions
        my_feature_group = FeatureGroup(name=self.output_uuid, sagemaker_session=self.sagemaker_session)
        my_feature_group.load_feature_definitions(data_frame=self.input_df)

        # Add some tags here
        tags = ['sageworks', 'public']

        # Create the Output Parquet file S3 Storage Path for this Feature Set
        s3_storage_path = f"{self.feature_set_s3_path}/{self.output_uuid}"

        # Write out the DataFrame to Parquet/FeatureSet/Athena
        my_feature_group.create(
            s3_uri=s3_storage_path,
            record_identifier_name=self.id_column,
            event_time_feature_name=self.event_time_column,
            # role_arn=get_execution_role(self.sagemaker_session),
            role_arn='arn:aws:iam::631406999566:role/service-role/AmazonSageMaker-ExecutionRole-20181215T180236',
            enable_online_store=True
        )

        # Ensure/wait for the feature group to be created
        self.ensure_feature_group_created(my_feature_group)

    def ensure_feature_group_created(self, feature_group):
        status = feature_group.describe().get("FeatureGroupStatus")
        while status == "Creating":
            self.log.info("FeatureSet being Created...")
            time.sleep(1)
            status = feature_group.describe().get("FeatureGroupStatus")
        self.log(f"FeatureSet {feature_group.name} successfully created")


# Simple test of the DFToFeatureSet functionality
def test():
    """Test the DFToFeatureSet Class"""

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Create some fake data
    fake_data = [
        {'id': 1, 'name': 'sue', 'age': 41, 'score': 7.8, 'date': datetime.utcnow()},
        {'id': 2, 'name': 'bob', 'age': 34, 'score': 6.4, 'date': datetime.utcnow()},
        {'id': 3, 'name': 'ted', 'age': 69, 'score': 8.2, 'date': datetime.utcnow()},
        {'id': 4, 'name': 'bill', 'age': 24, 'score': 5.3, 'date': datetime.utcnow()},
        {'id': 5, 'name': 'sally', 'age': 52, 'score': 9.5, 'date': datetime.utcnow()}
        ]
    fake_df = pd.DataFrame(fake_data)

    # Create my DF to Data Source Transform
    output_uuid = 'test-feature-set'
    df_to_features = DFToFeatureSet()
    df_to_features.set_input(fake_df, id_column='id', event_time_column='date')
    df_to_features.set_output_uuid(output_uuid)

    # Does my data pass validation?
    assert(df_to_features.validate_input())

    # Store this data into Athena/SageWorks
    df_to_features.transform()

    # Grab the output and query it for a dataframe
    output = df_to_features.get_output()
    df = output.query(f"select * from {output_uuid} limit 5")

    # Show the dataframe
    print(df)


if __name__ == "__main__":
    test()
