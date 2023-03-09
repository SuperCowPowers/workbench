"""TidyData: Example Class that demonstrates data cleanup for Light DataSources using Pandas"""
import logging

import awswrangler as wr
import json

# Local imports
from sageworks.utils.logging import logging_setup
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.transforms.transform_utils.data_to_pandas import DataToPandas
from sageworks.transforms.transform_utils.pandas_to_data import PandasToData

# Setup Logging
logging_setup()


class TidyData(Transform):
    def __init__(self):
        """TidyData: Class for filtering, sub-setting, and value constraints on Light DataSources uses Pandas"""

        # Call superclass init
        super().__init__()

        # Set up all my instance attributes
        self.input_type = TransformInput.DATA_SOURCE
        self.output_type = TransformOutput.DATA_SOURCE

    def transform(self):
        """Pull the input DataSource make sure it's 'clean' and output to a DataSource"""

        # Grab the Input (Data Source)
        input_df = DataToPandas(self.input_uuid).get_output()

        # Now Cleanup up the data and publish to the output location
        PandasToData(self.output_uuid)



    def transform(self, overwrite: bool = True, exclude_nans: bool = True):
        """Convert the CSV data into Parquet Format in the SageWorks S3 Bucket, and
           store the information about the data to the AWS Data Catalog sageworks database"""

        # Sanity Check for S3 Object size
        object_megabytes = self.input_size_mb()
        if object_megabytes > 100:
            self.log.error(f"S3 Object too big ({object_megabytes} MBytes): Use the S3BigToAthena class!")
            return

        # Add some tags here
        tags = ['sageworks', 'tidy_data']

        # Read in the S3 CSV as a Pandas DataFrame
        df = wr.s3.read_csv(self.s3_file_path, low_memory=False)

        # Create the Output Parquet file S3 Storage Path
        s3_storage_path = f"{self.data_source_s3_path}/{self.output_uuid}"

        # Write out the DataFrame to Parquet/DataStore/Athena
        wr.s3.to_parquet(df, path=s3_storage_path, dataset=True, mode='overwrite',
                         database=self.data_catalog_db, table=self.output_name,
                         description=f'SageWorks data source: {self.output_name}',
                         filename_prefix=f'{self.output_name}_',
                         parameters={'tags': json.dumps(tags)},
                         partition_cols=None)  # FIXME: Have some logic around partition columns


# Simple test of the TidyData functionality
def test():
    """Test the TidyData Class"""
    import pandas as pd

    # Create my Data Loader
    output_uuid = 'aqsol_data'
    my_loader = TidyData()
    my_loader.set_input('s3://sageworks-incoming-data/aqsol_public_data.csv')
    my_loader.set_output_name(output_uuid)

    # Does my S3 data exist?
    assert(my_loader.validate_input())

    # Store this data into Athena/SageWorks
    my_loader.transform()

    # Validate the output
    assert(my_loader.validate_output())

    # Grab the output and query it
    output = my_loader.get_output()
    query = f"select * from {output_uuid} limit 5"
    df = output.query(query)

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Show the dataframe
    print(df)


if __name__ == "__main__":
    test()
