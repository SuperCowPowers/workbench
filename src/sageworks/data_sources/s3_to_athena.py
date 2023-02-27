"""S3ToAthena: Class to move an S3 File into Athena"""
import awswrangler as wr
import json


class S3ToAthena:
    def __init__(self, name, resource_url: str):
        """S3ToAthena: Class to move an S3 File into Athena"""

        # S3 path if they want to store this data
        self.name = name
        self.resource_url = resource_url
        self.data_catalog_db = 'sageworks'
        self.data_source_s3_path = 's3://sageworks-data-sources'
        self.tags = set()

    def check(self) -> bool:
        """Does the S3 object exist"""
        print(f'Checking {self.name}...')
        return wr.s3.does_object_exist(self.resource_url)

    def load_into_athena(self, overwrite=True):
        """Convert the CSV data into Parquet Format in the SageWorks S3 Bucket, and
           store the information about the data to the AWS Data Catalog sageworks database"""
        s3_storage_path = f"{self.data_source_s3_path}/{self.name}"
        print('Storing into SageWorks...')

        # Add some tags here
        self.tags.add('sageworks')
        self.tags.add('public')

        # FIXME: Currently this pulls all the data down to the client (improve this later)
        df = wr.s3.read_csv(self.resource_url, low_memory=False)
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


# Simple test of the S3ToAthena functionality
def test():
    """Test the S3ToAthena Class"""

    # Create a Data Source
    my_data = S3ToAthena('aqsol_data', 's3://sageworks-incoming-data/aqsol_public_data.csv')

    # Does my S3 data exist?
    assert(my_data.check())

    # Store this data into Athena/SageWorks
    my_data.load_into_athena()

    # Now try out an Athena Query
    my_data.test_athena_query()


if __name__ == "__main__":
    test()
