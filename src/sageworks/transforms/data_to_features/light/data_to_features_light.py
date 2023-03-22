"""DataToFeaturesLight: Base Class for Light DataSource to FeatureSet using Pandas"""

# Local imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.transforms.pandas_transforms.data_to_pandas import DataToPandas
from sageworks.transforms.pandas_transforms.pandas_to_features import PandasToFeatures


class DataToFeaturesLight(Transform):
    def __init__(self, input_uuid: str = None, output_uuid: str = None):
        """DataToFeaturesLight: Base Class for Light DataSource to DataSource using Pandas"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.DATA_SOURCE
        self.output_type = TransformOutput.FEATURE_SET
        self.input_df = None
        self.output_df = None

    def pre_transform(self, **kwargs):
        """Pull the input DataSource into our Input Pandas DataFrame"""

        # Grab the Input (Data Source)
        self.input_df = DataToPandas(self.input_uuid).get_output()  # Shorthand for transform, get_output

    def post_transform(self, **kwargs):
        """At this point the output DataFrame should be populated, so publish it as a DataSource"""

        # Now publish to the output location
        output_features = PandasToFeatures()
        output_features.set_input(self.output_df, id_column='id')
        output_features.set_output_uuid(self.output_uuid)
        output_features.transform(**kwargs)


# Simple test of the DataToFeaturesLight functionality
def test():
    """Test the DataToFeaturesLight Class"""

    # My Test Class
    class MyTransform(DataToFeaturesLight):
        def __init__(self, input_uuid, output_uuid):
            super().__init__(input_uuid, output_uuid)

        def transform_impl(self, **kwargs):
            self.output_df = self.input_df

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = 'aqsol_data'
    output_uuid = 'test_aqsol_features'
    MyTransform(input_uuid, output_uuid).transform(delete_existing=True)


if __name__ == "__main__":
    test()
