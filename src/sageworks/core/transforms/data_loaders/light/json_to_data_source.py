"""JSONToDataSource: Class to move local JSON Files into a SageWorks DataSource"""

import os
import pandas as pd

# Local imports
from sageworks.core.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.core.transforms.pandas_transforms.pandas_to_data import PandasToData


class JSONToDataSource(Transform):
    """JSONToDataSource: Class to move local JSON Files into a SageWorks DataSource

    Common Usage:
        ```python
        json_to_data = JSONToDataSource(json_file_path, data_uuid)
        json_to_data.set_output_tags(["abalone", "json", "whatever"])
        json_to_data.transform()
        ```
    """

    def __init__(self, json_file_path: str, data_uuid: str):
        """JSONToDataSource: Class to move local JSON Files into a SageWorks DataSource

        Args:
            json_file_path (str): The path to the JSON file to be transformed
            data_uuid (str): The UUID of the SageWorks DataSource to be created
        """

        # Call superclass init
        super().__init__(json_file_path, data_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.LOCAL_FILE
        self.output_type = TransformOutput.DATA_SOURCE

    def transform_impl(self, overwrite: bool = True):
        """Convert the local JSON file into Parquet Format in the SageWorks Data Sources Bucket, and
        store the information about the data to the AWS Data Catalog sageworks database
        """

        # Report the transformation initiation
        json_file = os.path.basename(self.input_uuid)
        self.log.info(f"Starting {json_file} -->  DataSource: {self.output_uuid}...")

        # Read in the Local JSON as a Pandas DataFrame
        df = pd.read_json(self.input_uuid, lines=True)

        # Use the SageWorks Pandas to Data Source class
        pandas_to_data = PandasToData(self.output_uuid)
        pandas_to_data.set_input(df)
        pandas_to_data.set_output_tags(self.output_tags)
        pandas_to_data.add_output_meta(self.output_meta)
        pandas_to_data.transform()

        # Report the transformation results
        self.log.info(f"{json_file} -->  DataSource: {self.output_uuid} Complete!")

    def post_transform(self, **kwargs):
        """Post-Transform"""
        self.log.info("Post-Transform: S3 to DataSource...")

        # Note: We do not need to onboard because PandasToData already onboarded


if __name__ == "__main__":
    """Exercise the JSONToDataSource Class"""
    import sys
    from pathlib import Path

    # Get the path to the dataset in the repository data directory
    data_path = str(Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "test_data.json")

    # Create my Data Loader
    my_loader = JSONToDataSource(data_path, "test_data_json")
    my_loader.set_output_tags(["test", "small"])

    # Store this data as a SageWorks DataSource
    my_loader.transform()
