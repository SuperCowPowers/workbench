"""DataToPandas: Class to transform a Data Source into a Pandas DataFrame"""
import pandas as pd

# Local imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.artifacts.data_sources.athena_source import AthenaSource


class DataToPandas(Transform):
    """DataToPandas: Class to transform a Data Source into a Pandas DataFrame

    Common Usage:
        data_to_df = DataToPandas(data_source_uuid)
        data_to_df.transform(max_rows=<optional max rows to sample>)
        my_df = data_to_df.get_output()
    """

    def __init__(self, input_uuid: str):
        """DataToPandas Initialization"""

        # Call superclass init
        super().__init__(input_uuid, "DataFrame")

        # Set up all my instance attributes
        self.input_type = TransformInput.DATA_SOURCE
        self.output_type = TransformOutput.PANDAS_DF
        self.output_df = None
        self.transform_run = False

    def transform_impl(self, max_rows=100000):
        """Convert the DataSource into a Pandas DataFrame"""

        # Grab the Input (Data Source)
        input_data = AthenaSource(self.input_uuid)
        if not input_data.exists():
            self.log.critical(f"Data Check on {self.input_uuid} failed!")
            return

        # Get the number of rows in the DataSource
        num_rows = input_data.num_rows()

        # If the data source has more rows than max_rows, do a sample query
        if num_rows > max_rows:
            percentage = round(max_rows * 100.0 / num_rows)
            self.log.warning(f"DataSource has {num_rows} rows.. sampling down to {max_rows}...")
            query = f"SELECT * FROM {self.input_uuid} TABLESAMPLE BERNOULLI({percentage})"
        else:
            query = f"SELECT * FROM {self.input_uuid}"

        # Mark the transform as complete and set the output DataFrame
        self.transform_run = True
        self.output_df = input_data.query(query)

    def post_transform(self, **kwargs):
        """Post-Transform: Any checks on the Pandas DataFrame that need to be done"""
        self.log.info("Post-Transform: Checking Pandas DataFrame...")
        self.log.info(f"DataFrame Shape: {self.output_df.shape}")

    def get_output(self) -> pd.DataFrame:
        """Get the DataFrame Output from this Transform"""
        if not self.transform_run:
            self.transform()
        return self.output_df


if __name__ == "__main__":
    """Exercise the DataToPandas Class"""

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 15)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Grab a Data Source
    data_uuid = "test_data"

    # Create the DataSource to DF Transform
    data_to_df = DataToPandas(data_uuid)

    # Transform the DataSource into a Pandas DataFrame (with max_rows = 1000)
    data_to_df.transform(max_rows=1000)

    # Grab the output dataframe and show it
    my_df = data_to_df.get_output()
    print(my_df)
