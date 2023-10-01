"""DataToFeaturesLight: Base Class for Light DataSource to FeatureSet using Pandas"""

# Local imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.transforms.pandas_transforms.data_to_pandas import DataToPandas
from sageworks.transforms.pandas_transforms.pandas_to_features import PandasToFeatures


class DataToFeaturesLight(Transform):
    """DataToFeaturesLight: Base Class for Light DataSource to FeatureSet using Pandas

    Common Usage:
        to_features = DataToFeaturesLight(data_uuid, feature_uuid)
        to_features.set_output_tags(["abalone", "public", "whatever"])
        to_features.transform(id_column="id"/None, event_time_column="date"/None, query=str/None)
    """

    def __init__(self, data_uuid: str, feature_uuid: str):
        """DataToFeaturesLight Initialization"""

        # Call superclass init
        super().__init__(data_uuid, feature_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.DATA_SOURCE
        self.output_type = TransformOutput.FEATURE_SET
        self.input_df = None
        self.output_df = None
        self.target = None

    def pre_transform(self, **kwargs):
        """Pull the input DataSource into our Input Pandas DataFrame"""

        # Grab the Input (Data Source)
        self.input_df = DataToPandas(self.input_uuid).get_output()  # Shorthand for transform, get_output

    def transform_impl(self, target: str = None, query: str = None, column_select: list = None, **kwargs):
        """Optional Query to filter the input DataFrame, then publish to the output location
        Args:
            target(str): The name of the target column
            query(str): Optional query to filter the input DataFrame
            column_select(list): Optional list of columns to select from the input DataFrame
        Notes:
            Query is a Pandas Expression, e.g. 'col1<2.5 & col2=="x"'
            Column Select is a list of column names, e.g. ['col3', 'col4']
        """
        self.target = target
        self.output_df = self.input_df.query(query).reset_index(drop=True) if query else self.input_df
        self.output_df = self.output_df[column_select] if column_select else self.output_df

    def post_transform(self, id_column=None, event_time_column=None, **kwargs):
        """At this point the output DataFrame should be populated, so publish it as a Feature Set"""
        # Now publish to the output location
        output_features = PandasToFeatures(self.output_uuid)
        output_features.set_input(self.output_df, self.target, id_column=id_column, event_time_column=event_time_column)
        output_features.set_output_tags(self.output_tags)
        output_features.add_output_meta(self.output_meta)
        output_features.transform()


if __name__ == "__main__":
    """Exercise the DataToFeaturesLight Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "test_data"
    output_uuid = "test_feature_set"
    data_to_features = DataToFeaturesLight(input_uuid, output_uuid)
    data_to_features.set_output_tags(["test", "small"])
    data_to_features.transform(id_column="id", event_time_column="date")
