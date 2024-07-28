"""Class: DataCatalog Helper Class for the AWS Data Catalog Databases"""

import sys
import argparse
import awswrangler as wr

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.connector import Connector
from sageworks.utils.aws_utils import compute_size


class DataCatalog(Connector):
    """DataCatalog: Helper Class for the AWS Data Catalog Databases"""

    def __init__(self, database_scope: (str, list) = "sageworks"):
        """DataCatalog: Helper Class for the AWS Data Catalog
        Args: -
            database_scope (str, list) - The scope of the databases to query (default = 'sageworks',
                                         can be list ['a', 'b'], also 'all' for account wide scope)
        """
        # Call SuperClass Initialization
        super().__init__()

        # Set up our database scope
        self.database_scope = database_scope if isinstance(database_scope, list) else [database_scope]

        # Set up our internal data storage
        self.data_catalog_metadata = {}

    def check(self) -> bool:
        """Check if we can reach/connect to this AWS Service"""
        try:
            wr.catalog.get_databases(boto3_session=self.boto_session)
            return True
        except Exception as e:
            self.log.critical(f"Error connecting to AWS Data Catalog: {e}")
            return False

    def refresh(self):
        """Refresh all the tables in all the catalog databases"""
        for database in self.database_scope:
            # Get all table metadata from the Glue catalog Database
            self.log.debug(f"Refreshing Data Catalog Database: {database}...")
            all_tables = wr.catalog.get_tables(database=database, boto3_session=self.boto_session)

            # Convert to a data structure with direct lookup
            self.data_catalog_metadata[database] = {table["Name"]: table for table in all_tables}

            # Track the size of the metadata
            for key in self.data_catalog_metadata[database].keys():
                self.metadata_size_info[database][key] = compute_size(self.data_catalog_metadata[database][key])

    def summary(self) -> dict:
        """Return a summary of all the tables in this AWS Data Catalog Database"""
        return self.data_catalog_metadata

    def get_tables(self, database: str) -> list:
        """Get all the table names in this database

        Args:
            database (str): The name of the database
        Returns:
            list: List of table names
        """
        return list(self.data_catalog_metadata[database].keys())


if __name__ == "__main__":
    """Exercises the DataCatalog Class"""
    from pprint import pprint

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print("Unrecognized args: %s" % commands)
        sys.exit(1)

    # Create the class and get the AWS Data Catalog database info
    catalog = DataCatalog(["sageworks", "sagemaker_featurestore"])
    catalog.refresh()

    # Get the Summary Information
    pprint(catalog.summary())

    # Print out the metadata sizes for this connector
    pprint(catalog.get_metadata_sizes())
