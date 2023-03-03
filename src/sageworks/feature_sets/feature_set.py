"""FeatureSet: SageWork Feature Set accessible through Athena"""
import pandas as pd
import awswrangler as wr
import logging

# Local Imports
from sageworks.data_sources.athena_source import AthenaSource
from sageworks.meta.sageworks_meta import MetaCategory, SageWorksMeta
from sageworks.utils.logging import logging_setup

# Setup Logging
logging_setup()


class FeatureSet(AthenaSource):
    """FeatureSet: SageWork Feature Set accessible through Athena"""
    def __init__(self, feature_set_name):
        """FeatureSet Initialization

        Args:
            feature_set_name (str): Name of Athena Table in the SageWorks Database
        """
        self.feature_set_name = feature_set_name
        self.log = logging.getLogger(__name__)

        # Call SuperClass (AthenaSource) Initialization
        super().__init__(feature_set_name)

        # Grab a SageWorks Metadata object and pull information for Feature Sets
        self.sage_meta = SageWorksMeta()
        self.my_meta = self.sage_meta.get(MetaCategory.FEATURE_SETS).get(self.feature_set_name)

        # All done
        print(f'FeatureSet Initialized: {feature_set_name}')

    def check(self) -> bool:
        """Does the feature_set_name exist in the SageWorks Metadata?"""
        if self.my_meta is None:
            self.log.critical(f'FeatureSet.check() {self.feature_set_name} not found in SageWorks Metadata!')
            return False
        return True

    def get_num_rows(self) -> int:
        """Return the number of rows for this Data Source"""
        count_df = self.query(f"select count(*) AS count from {self.feature_set_name}")
        return count_df['count'][0]

    def get_num_columns(self) -> int:
        """Return the number of columns for this Data Source"""
        if self.column_names is None:
            self.get_column_names()
        self.num_columns = len(self.column_names)
        return self.num_columns

    def get_column_names(self) -> list[str]:
        """Return the column names for this Athena Table"""
        if self.column_names is None:
            self.column_names = [item['Name'] for item in self.my_meta['StorageDescriptor']['Columns']]
        return self.column_names

    def query(self, query: str) -> pd.DataFrame:
        """Query the FeatureSet"""
        df = wr.athena.read_sql_query(sql=query, database=self.data_catalog_db)
        scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
        self.log.info(f"Athena Query successful (scanned bytes: {scanned_bytes})")
        return df

    def schema_validation(self):
        """All data sources will have some form of schema validation.
           For things like RDS tables this might just be a return True
           but for other data source types there should be some code that
           looks at the columns and types and does something reasonable
           """
        return True

    def data_quality_review(self):
        """All data sources should have some form of data quality review.
           After schema_validation the class should 'inspect' the data values.
           - What percentage of values in each column are NaNs/Null?
           - Are there only a few unique values in each column (low variance?)
           - Are there 'crazy' values way out of 3Sigma range?
           - TBD
           Given that plots and charts will be super helpful for this the data
           quality review process should probably involve a web page and click 'OK' button
           Having a Data Quality Web page is something we want to do anyway
           """
        return True

    def generate_feature_set(self, feature_type: str) -> bool:
        """Concrete Classes will support different feature set generations"""
        return True


# Simple test of the FeatureSet functionality
def test():
    """Test for FeatureSet Class"""

    # Grab a FeatureSet object and pull some information from it
    my_features = FeatureSet('AqSolDB-base')

    # Call the various methods

    # Does this Athena data exist?
    assert(my_features.check())

    # How many rows and columns?
    rows = my_features.get_num_rows()
    columns = my_features.get_num_columns()
    print(f'Rows: {rows} Columns: {columns}')

    # What are the column names?
    print(my_features.get_column_names())

    # Run a query to only pull back only some data
    query = f"select id, name, smiles from {my_features.feature_set_name} limit 10"
    df = my_features.query(query)
    print(df)


if __name__ == "__main__":
    test()
