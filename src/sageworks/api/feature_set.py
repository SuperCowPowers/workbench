"""FeatureSet: SageWorks FeatureSet API Class"""

# SageWorks Imports
from sageworks.core.artifacts.feature_set_core import FeatureSetCore
from sageworks.core.transforms.features_to_model.features_to_model import FeaturesToModel
from sageworks.core.artifacts.model_core import ModelType
from sageworks.api.model import Model


class FeatureSet(FeatureSetCore):
    """FeatureSet: SageWorks FeatureSet API Class

    Common Usage:
        my_features = FeatureSet(name)
        my_features.summary()
        my_features.details()
        my_features.to_model()
    """

    def __init__(self, name):
        """FeatureSet Initialization
        Args:
            name (str): The name of the FeatureSet
        """
        # Call superclass init
        super().__init__(name)

    def to_model(self, model_type: ModelType, target_column: str = None, name: str = None, tags: list = None, description: str = None):
        """Create a Model from the FeatureSet
        Args:
            model_type (ModelType): The type of model to create (See ModelType)
            target_column (str): The target column for the model (optional)
            name (str): Set the name for the model (optional)
            tags (list): Set the tags for the model (optional)
            description (str): Set the description for the model (optional)
        Returns:
            Model: The Model created from the FeatureSet
        """

        # Create the Model Name and Tags
        model_name = self.uuid.replace("_features", "").replace("_", "-") + "-model" if name is None else name
        tags = [model_name] if tags is None else tags

        # Transform the FeatureSet into a Model
        features_to_model = FeaturesToModel(self.uuid, model_name, model_type=model_type)
        features_to_model.set_output_tags(tags)
        features_to_model.transform(target_column=target_column, description=description)

        # Return the Model
        return Model(model_name)


if __name__ == "__main__":
    """Exercise the FeatureSet Class"""
    from pprint import pprint

    # Retrieve an existing Data Source
    my_features = FeatureSet("test_features")
    pprint(my_features.summary())
    pprint(my_features.details())

    # Create a Model from the FeatureSet
    my_model = my_features.to_model(model_type=ModelType.REGRESSOR, target_column="iq_score")
