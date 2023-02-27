"""S3SmallToAthena: Class to move SMALL S3 Files into Athena"""
import awswrangler as wr
import json
import logging

# Local imports
from sageworks.utils.logging import logging_setup

# Setup Logging
logging_setup()


class S3SmallToAthena:
    def __init__(self, name, s3_file_path: str):
        """S3SmallToAthena: Class to move SMALL S3 Files into Athena"""

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

    def object_size_mb(self) -> int:
        """ Get the size of the S3 object in MBytes"""
        size_in_bytes = wr.s3.size_objects(self.s3_file_path)[self.s3_file_path]
        size_in_mb = int(size_in_bytes/1_000_000)
        return size_in_mb

    def load_into_athena(self, overwrite=True):
        """Convert the CSV data into Parquet Format in the SageWorks S3 Bucket, and
           store the information about the data to the AWS Data Catalog sageworks database"""

        # Sanity Check for S3 Object size
        object_megabytes = self.object_size_mb()
        if object_megabytes > 100:
            self.log.error(f"S3 Object too big ({object_megabytes} MBytes): Use the S3BigToAthena class!")
            return

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


# Simple test of the S3SmallToAthena functionality
def test():
    """Test the S3SmallToAthena Class"""

    # Create a Data Source
    my_loader = S3SmallToAthena('aqsol_data', 's3://sageworks-incoming-data/aqsol_public_data.csv')

    # Does my S3 data exist?
    assert(my_loader.check())

    # Store this data into Athena/SageWorks
    my_loader.load_into_athena()

    # Now try out an Athena Query
    my_loader.test_athena_query()


if __name__ == "__main__":
    test()
