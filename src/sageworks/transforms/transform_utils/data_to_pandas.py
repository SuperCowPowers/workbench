"""DataSourceToPandas: Class to transform a Data Source into a Pandas DataFrame"""
import pandas as pd

# Local imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.artifacts.data_sources.athena_source import AthenaSource


class DataSourceToPandas(Transform):
    def __init__(self):
        """DataSourceToPandas: Class to transform a Data Source into a Pandas DataFrame"""

        # Call superclass init
        super().__init__()

        # Set up all my class instance vars
        self.input_type = TransformInput.DATA_SOURCE
        self.output_type = TransformInput.PANDAS_DF
        self.data_catalog_db = 'sageworks'
        self.output_df = None

    def get_dataframe(self) -> pd.DataFrame:
        """Get the DataFrame Output from this Transform"""
        return self.output_df

    def transform(self, max_rows=100000):
        """Convert the DataSource into a Pandas DataFrame"""

        # Grab the Input (Data Source)
        input_data = AthenaSource(self.data_catalog_db, self.input_uuid)
        if not input_data.check():
            self.log.critical(f"Data Check on {self.input_uuid} failed!")
            return

        # Get the number of rows in the DataSource
        num_rows = input_data.num_rows()

        # If the data source has more rows than max_rows, do a sample query
        if num_rows > max_rows:
            percentage = round(max_rows*100.0/num_rows)
            self.log.warning(f"DataSource has {num_rows} rows.. sampling down to {max_rows}...")
            query = f"SELECT * FROM {self.input_uuid} TABLESAMPLE BERNOULLI({percentage})"
        else:
            query = f"SELECT * FROM {self.input_uuid}"
        self.output_df = input_data.query(query)


# Simple test of the DataSourceToPandas functionality
def test():
    """Test the DataSourceToPandas Class"""

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Grab a Data Source
    data_uuid = 'aqsol_data'

    # Create the DataSource to DF Transform
    data_to_df = DataSourceToPandas()
    data_to_df.set_input_uuid(data_uuid)

    # Transform the DataSource into a Pandas DataFrame (with max_rows = 1000)
    data_to_df.transform(max_rows=1000)

    # Grab the output dataframe and show it
    my_df = data_to_df.get_dataframe()
    print(my_df)


if __name__ == "__main__":
    test()
