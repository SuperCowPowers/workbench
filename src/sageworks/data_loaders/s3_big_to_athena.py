"""S3BigToAthena: Class to move BIG S3 Files into Athena"""
import awswrangler as wr
import json
import logging

# FIXME: This will be a glue jobs that uses SPARK to convert CSV to Parquet in a 
#        scalable say and then populate the AWS Data Catalog with the information

# Local imports
from sageworks.utils.logging import logging_setup

# Setup Logging
logging_setup()


class S3BigToAthena:
    def __init__(self, name, s3_file_path: str):
        """S3BigToAthena: Class to move BIG S3 Files into Athena"""

        # Set up all my class instance vars
        self.log = logging.getLogger(__name__)
        self.name = name
        self.s3_file_path = s3_file_path
        self.data_catalog_db = 'sageworks'
        self.data_source_s3_path = 's3://sageworks-data-sources'
        self.tags = set()

    def check(self) -> bool:
        """Does the S3 object exist"""
        print(f'Checking {self.name}...')
        return wr.s3.does_object_exist(self.s3_file_path)

    def load_into_athena(self, overwrite=True):
        """Convert the CSV data into Parquet Format in the SageWorks S3 Bucket, and
           store the information about the data to the AWS Data Catalog sageworks database"""

        # FIXME: Spark job to convert CSV to Parquet Here
        # FIXME: Populate the AWS Data Catalog with the information Here

        # Okay going to load this into Athena/Parquet
        s3_storage_path = f"{self.data_source_s3_path}/{self.name}"
        print('Storing into SageWorks...')

        # Add some tags here
        self.tags.add('sageworks')
        self.tags.add('public')

        # FIXME: Put in a size limit sanity check here
        df = wr.s3.read_csv(self.s3_file_path, low_memory=False)
        wr.s3.to_parquet(df, path=s3_storage_path, dataset=True, mode='overwrite',
                         database=self.data_catalog_db, table=self.name,
                         description=f'SageWorks data source: {self.name}',
                         filename_prefix=f'{self.name}_',
                         parameters={'tags': json.dumps(list(self.tags))},
                         partition_cols=None)  # FIXME: Have some logic around partition columns

    def test_athena_query(self):
        """Run AFTER a data source has been stored (with _store_in_sageworks) this method
           insures that everything went well and the data can be queried from Athena"""
        query = f"select count(*) as count from {self.name} limit 1"
        df = wr.athena.read_sql_query(sql=query, database=self.data_catalog_db)
        print(df.head())
        scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
        print(f"Athena Query successful (scanned bytes: {scanned_bytes})")


# Simple test of the S3BigToAthena functionality
def test():
    """Test the S3BigToAthena Class"""

    # Create a Data Source
    my_loader = S3BigToAthena('aqsol_data', 's3://sageworks-incoming-data/aqsol_public_data.csv')

    # Does my S3 data exist?
    assert(my_loader.check())

    # Store this data into Athena/SageWorks
    my_loader.load_into_athena()

    # Now try out an Athena Query
    my_loader.test_athena_query()


if __name__ == "__main__":
    test()
