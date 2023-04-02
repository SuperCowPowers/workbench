"""DataToFeaturesLight: Base Class for Light DataSource to FeatureSet using Pandas"""

# Local imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.transforms.pandas_transforms.data_to_pandas import DataToPandas
from sageworks.transforms.pandas_transforms.pandas_to_features import PandasToFeatures


class DataToFeaturesLight(Transform):
    def __init__(self, input_uuid: str, output_uuid: str):
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

    def transform_impl(self, **kwargs):
        """Base Class is simply an identity transform"""
        self.output_df = self.input_df

    def post_transform(self, id_column=None, event_time_column=None, delete_existing=True):
        """At this point the output DataFrame should be populated, so publish it as a Feature Set"""

        # Now publish to the output location
        output_features = PandasToFeatures(self.output_uuid)
        output_features.set_input(self.output_df, id_column=id_column, event_time_column=event_time_column)
        output_features.set_output_tags(self.output_tags)
        output_features.set_output_meta(self.output_meta)
        output_features.transform(delete_existing=delete_existing)


if __name__ == "__main__":
    """Exercise the DataToFeaturesLight Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = 'test_data'
    output_uuid = 'test_feature_set'
    data_to_features = DataToFeaturesLight(input_uuid, output_uuid)
    data_to_features.set_output_tags(['test', 'small'])
    data_to_features.set_output_meta({'sageworks_input': input_uuid})
    data_to_features.transform(id_column='id', event_time_column='date')
