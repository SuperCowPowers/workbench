"""DataToDataLight: Base Class for Light DataSource to DataSource using Pandas"""

# Local imports
from sageworks.core.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.core.transforms.pandas_transforms.data_to_pandas import DataToPandas
from sageworks.core.transforms.pandas_transforms.pandas_to_data import PandasToData


class DataToDataLight(Transform):
    """DataToDataLight: Base Class for Light DataSource to DataSource using Pandas

    Common Usage:
        ```python
        to_data = DataToDataLight(input_data_uuid, output_data_uuid)
        to_data.set_output_tags(["abalone", "public", "whatever"])
        to_data.transform(query=<optional SQL query to filter/process data>)

        Note: query is the best way to use this class, so use it :)
        ```
    """

    def __init__(self, input_data_uuid: str, output_data_uuid: str):
        """DataToDataLight Initialization"""

        # Call superclass init
        super().__init__(input_data_uuid, output_data_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.DATA_SOURCE
        self.output_type = TransformOutput.DATA_SOURCE
        self.input_df = None
        self.output_df = None

    def pre_transform(self, query: str = None):
        """Pull the input DataSource into our Input Pandas DataFrame
        Args:
            query(str): Optional query to filter the input DataFrame
        """

        # Grab the Input (Data Source)
        data_to_pandas = DataToPandas(self.input_uuid)
        data_to_pandas.transform(query=query)
        self.input_df = data_to_pandas.get_output()

    def transform_impl(self, **kwargs):
        """Base Class simply takes the input DataFrame and copies it to the output DataFrame"""
        self.output_df = self.input_df

    def post_transform(self, **kwargs):
        """At this point the output DataFrame should be populated, so publish it as a DataSource"""

        # Now publish to the output location
        output_data_source = PandasToData(self.output_uuid)
        output_data_source.set_input(self.output_df)
        output_data_source.set_output_tags(self.output_tags)
        output_data_source.add_output_meta(self.output_meta)
        output_data_source.transform(**kwargs)


if __name__ == "__main__":
    """Exercise the DataToDataLight Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "abalone_data"
    output_uuid = "abalone_data_copy"
    data_to_data = DataToDataLight(input_uuid, output_uuid)
    data_to_data.set_output_tags(["abalone", "public"])
    data_to_data.transform(query=f"SELECT * from {input_uuid} limit 100")
