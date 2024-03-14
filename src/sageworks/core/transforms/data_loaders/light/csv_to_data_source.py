"""CSVToDataSource: Class to move local CSV Files into a SageWorks DataSource"""

import os
import pandas as pd

# Local imports
from sageworks.core.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.core.transforms.pandas_transforms.pandas_to_data import PandasToData
from sageworks.utils.pandas_utils import convert_object_columns


class CSVToDataSource(Transform):
    """CSVToDataSource: Class to move local CSV Files into a SageWorks DataSource

    Common Usage:
        ```
        csv_to_data = CSVToDataSource(csv_file_path, data_uuid)
        csv_to_data.set_output_tags(["abalone", "csv", "whatever"])
        csv_to_data.transform()
        ```
    """

    def __init__(self, csv_file_path: str, data_uuid: str):
        """CSVToDataSource: Class to move local CSV Files into a SageWorks DataSource

        Args:
            csv_file_path (str): The path to the CSV file to be transformed
            data_uuid (str): The UUID of the SageWorks DataSource to be created
        """

        # Call superclass init
        super().__init__(csv_file_path, data_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.LOCAL_FILE
        self.output_type = TransformOutput.DATA_SOURCE

    def transform_impl(self, overwrite: bool = True):
        """Convert the local CSV file into Parquet Format in the SageWorks Data Sources Bucket, and
        store the information about the data to the AWS Data Catalog sageworks database
        """

        # Report the transformation initiation
        csv_file = os.path.basename(self.input_uuid)
        self.log.info(f"Starting {csv_file} -->  DataSource: {self.output_uuid}...")

        # Read in the Local CSV as a Pandas DataFrame
        df = pd.read_csv(self.input_uuid, low_memory=False)
        df = convert_object_columns(df)

        # Use the SageWorks Pandas to Data Source class
        pandas_to_data = PandasToData(self.output_uuid)
        pandas_to_data.set_input(df)
        pandas_to_data.set_output_tags(self.output_tags)
        pandas_to_data.add_output_meta(self.output_meta)
        pandas_to_data.transform()

        # Report the transformation results
        self.log.info(f"{csv_file} -->  DataSource: {self.output_uuid} Complete!")

    def post_transform(self, **kwargs):
        """Post-Transform"""
        self.log.info("Post-Transform: S3 to DataSource...")

        # Note: We do not need to onboard because PandasToData already onboarded


if __name__ == "__main__":
    """Exercise the CSVToDataSource Class"""
    import sys
    from pathlib import Path

    # Get the path to the dataset in the repository data directory
    data_path = str(Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "abalone.csv")

    # Create my Data Loader
    my_loader = CSVToDataSource(data_path, "abalone_data")
    my_loader.set_output_tags(["abalone", "public"])

    # Store this data as a SageWorks DataSource
    my_loader.transform()
