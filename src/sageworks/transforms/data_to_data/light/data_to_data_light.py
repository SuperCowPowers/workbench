"""DataToDataLight: Base Class for Light DataSource to DataSource using Pandas"""

# Local imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.transforms.pandas_transforms.data_to_pandas import DataToPandas
from sageworks.transforms.pandas_transforms.pandas_to_data import PandasToData


class DataToDataLight(Transform):
    def __init__(self, input_uuid: str = None, output_uuid: str = None):
        """DataToDataLight: Base Class for Light DataSource to DataSource using Pandas"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.DATA_SOURCE
        self.output_type = TransformOutput.DATA_SOURCE
        self.input_df = None
        self.output_df = None

    def pre_transform(self, **kwargs):
        """Pull the input DataSource into our Input Pandas DataFrame"""

        # Grab the Input (Data Source)
        self.input_df = DataToPandas(self.input_uuid).get_output()  # Shorthand for transform, get_output

    def transform_impl(self, **kwargs):
        """Base Class is simply an identity transform"""
        self.output_df = self.input_df

    def post_transform(self, **kwargs):
        """At this point the output DataFrame should be populated, so publish it as a DataSource"""

        # Now publish to the output location
        output_data_source = PandasToData(self.output_uuid)
        output_data_source.set_input(self.output_df)
        output_data_source.transform()


# Simple test of the DataToDataLight functionality
def test():
    """Test the DataToDataLight Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = 'abalone_data'
    output_uuid = 'abalone_data_copy'
    DataToDataLight(input_uuid, output_uuid).transform()


if __name__ == "__main__":
    test()
