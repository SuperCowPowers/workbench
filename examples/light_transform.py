"""LightTransform: Example Class that demonstrates a Transform for Light DataSources using Pandas"""

# Local imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.transforms.pandas_transforms.data_to_pandas import DataToPandas
from sageworks.transforms.pandas_transforms.pandas_to_data import PandasToData


class LightTransform(Transform):
    def __init__(self, input_uuid: str = None, output_uuid: str = None):
        """"LightTransform: Example Class that demonstrates a Transform for Light DataSources using Pandas"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.DATA_SOURCE
        self.output_type = TransformOutput.DATA_SOURCE

    def transform(self):
        """Pull the input DataSource, do something kewl, and output to a DataSource"""

        # Grab the Input (Data Source)
        input_df = DataToPandas(self.input_uuid).get_output()  # Shorthand for transform, get_output

        # Do something interesting with the Pandas DataFrame
        interesting_df = input_df  # Add kewl stuff here

        # Now publish to the output location
        output_data = PandasToData(self.output_uuid)
        output_data.set_input(interesting_df)
        output_data.transform()


if __name__ == "__main__":
    """Run/Test the LightTransform Class"""
    from sageworks.artifacts.data_sources.athena_source import AthenaSource

    # Create the class with inputs and outputs and invoke the transform
    my_input = 'aqsol_data'
    my_output = 'aqsol_data_clean'
    LightTransform(my_input, my_output).transform()

    # Use SageWorks to grab the output and query it for a dataframe
    output = AthenaSource(my_output)
    df = output.query(f"select * from {my_output} limit 5")

    # Show the dataframe
    print(df)
