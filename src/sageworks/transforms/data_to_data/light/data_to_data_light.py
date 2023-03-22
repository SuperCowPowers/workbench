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

    def post_transform(self, **kwargs):
        """At this point the output DataFrame should be populated, so publish it as a DataSource"""

        # Now publish to the output location
        output_data_source = PandasToData(self.output_uuid)
        output_data_source.set_input(self.output_df)
        output_data_source.transform()


# Simple test of the DataToDataLight functionality
def test():
    """Test the DataToDataLight Class"""
    import pandas as pd
    from sageworks.artifacts.data_sources.athena_source import AthenaSource

    # My Test Class
    class MyTransform(DataToDataLight):
        def __init__(self, input_uuid, output_uuid):
            super().__init__(input_uuid, output_uuid)

        def transform_impl(self):
            self.output_df = self.input_df

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = 'aqsol_data'
    output_uuid = 'aqsol_data_clean'
    MyTransform(input_uuid, output_uuid).transform()

    # Grab the output and query it for a dataframe
    output = AthenaSource(output_uuid)
    df = output.query(f"select * from {output_uuid} limit 5")

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Show the dataframe
    print(df)


if __name__ == "__main__":
    test()
