"""DataToPandas: Class to transform a Data Source into a Pandas DataFrame"""

import pandas as pd

# Local imports
from sageworks.core.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.core.artifacts.data_source_factory import DataSourceFactory


class DataToPandas(Transform):
    """DataToPandas: Class to transform a Data Source into a Pandas DataFrame

    Common Usage:
        ```
        data_to_df = DataToPandas(data_source_uuid)
        data_to_df.transform(query=<optional SQL query to filter/process data>)
        data_to_df.transform(max_rows=<optional max rows to sample>)
        my_df = data_to_df.get_output()

        Note: query is the best way to use this class, so use it :)
        ```
    """

    def __init__(self, input_uuid: str):
        """DataToPandas Initialization"""

        # Call superclass init
        super().__init__(input_uuid, "DataFrame")

        # Set up all my instance attributes
        self.input_type = TransformInput.DATA_SOURCE
        self.output_type = TransformOutput.PANDAS_DF
        self.output_df = None

    def transform_impl(self, query: str = None, max_rows=100000):
        """Convert the DataSource into a Pandas DataFrame
        Args:
            query(str): The query to run against the DataSource (default: None)
            max_rows(int): The maximum number of rows to return (default: 100000)
        """

        # Grab the Input (Data Source)
        input_data = DataSourceFactory(self.input_uuid)
        if not input_data.exists():
            self.log.critical(f"Data Check on {self.input_uuid} failed!")
            return

        # If a query is provided, that overrides the queries below
        if query:
            self.log.info(f"Querying {self.input_uuid} with {query}...")
            self.output_df = input_data.query(query)
            return

        # If the data source has more rows than max_rows, do a sample query
        num_rows = input_data.num_rows()
        if num_rows > max_rows:
            percentage = round(max_rows * 100.0 / num_rows)
            self.log.important(f"DataSource has {num_rows} rows.. sampling down to {max_rows}...")
            query = f"SELECT * FROM {self.input_uuid} TABLESAMPLE BERNOULLI({percentage})"
        else:
            query = f"SELECT * FROM {self.input_uuid}"

        # Mark the transform as complete and set the output DataFrame
        self.output_df = input_data.query(query)

    def post_transform(self, **kwargs):
        """Post-Transform: Any checks on the Pandas DataFrame that need to be done"""
        self.log.info("Post-Transform: Checking Pandas DataFrame...")
        self.log.info(f"DataFrame Shape: {self.output_df.shape}")

    def get_output(self) -> pd.DataFrame:
        """Get the DataFrame Output from this Transform"""
        return self.output_df


if __name__ == "__main__":
    """Exercise the DataToPandas Class"""

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 15)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Grab a Data Source
    data_uuid = "abalone_data"

    # Create the DataSource to DF Transform
    data_to_df = DataToPandas(data_uuid)

    # Transform the DataSource into a Pandas DataFrame (with max_rows = 1000)
    data_to_df.transform(max_rows=1000)

    # Grab the output dataframe and show it
    my_df = data_to_df.get_output()
    print(my_df)

    # Now test the query functionality
    data_to_df.transform(query=f"SELECT * from {data_uuid} limit 100")
    my_df = data_to_df.get_output()
    print(my_df)
